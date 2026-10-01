import os
from datetime import datetime, timedelta
from typing import List, Optional
import urllib.parse
import jwt
from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import shutil
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# --- CONFIGURATION BASE DE DONNÉES (Compatible SQLite / PostgreSQL) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./batipilot.db")

# Fix pour la compatibilité Render/Heroku avec PostgreSQL (postgres:// -> postgresql://)[cite: 11, 12]
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- CONFIGURATION SÉCURITÉ & JWT ---
SECRET_KEY = os.getenv("SECRET_KEY", "cle_secrete_temporaire_batipilot_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 heures[cite: 11, 12]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# --- MODÈLES BASE DE DONNÉES (Multi-tenancy via entreprise_id) ---

class EntrepriseDB(Base):
    __tablename__ = "entreprises"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    mot_de_passe_hache = Column(String, nullable=False)

    chantiers = relationship("ChantierDB", back_populates="entreprise")


class ChantierDB(Base):
    __tablename__ = "chantiers"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False)
    client = Column(String, nullable=False)
    avancement = Column(Integer, default=0)
    marche = Column(Float, nullable=False)
    cout_reel = Column(Float, default=0.0)
    entreprise_id = Column(Integer, ForeignKey("entreprises.id"), nullable=False)

    entreprise = relationship("EntrepriseDB", back_populates="chantiers")
    soustraitants = relationship("SousTraitantDB", back_populates="chantier", cascade="all, delete-orphan")
    factures = relationship("FactureDB", back_populates="chantier", cascade="all, delete-orphan")


class SousTraitantDB(Base):
    __tablename__ = "soustraitants"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False)
    metier = Column(String, nullable=False)
    montant = Column(Float, nullable=False)
    urssaf_ok = Column(Boolean, default=True)
    chantier_id = Column(Integer, ForeignKey("chantiers.id"), nullable=False)
    entreprise_id = Column(Integer, ForeignKey("entreprises.id"), nullable=False)

    chantier = relationship("ChantierDB", back_populates="soustraitants")


class FactureDB(Base):
    __tablename__ = "factures"

    id = Column(Integer, primary_key=True, index=True)
    num = Column(String, index=True, nullable=False)
    montant = Column(Float, nullable=False)
    payee = Column(Boolean, default=False)
    chantier_id = Column(Integer, ForeignKey("chantiers.id"), nullable=False)
    entreprise_id = Column(Integer, ForeignKey("entreprises.id"), nullable=False)

    chantier = relationship("ChantierDB", back_populates="factures")


Base.metadata.create_all(bind=engine)


# --- SCHÉMAS PYDANTIC ---

class EntrepriseCreate(BaseModel):
    nom: str
    email: EmailStr
    mot_de_passe: str

class EntrepriseResponse(BaseModel):
    id: int
    nom: str
    email: str
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChantierCreate(BaseModel):
    nom: str
    client: str
    marche: float
    cout_reel: Optional[float] = 0.0

class ChantierResponse(ChantierCreate):
    id: int
    avancement: int
    class Config:
        from_attributes = True

class SousTraitantCreate(BaseModel):
    nom: str
    metier: str
    montant: float
    chantier_id: int
    urssaf_ok: Optional[bool] = True

class SousTraitantResponse(SousTraitantCreate):
    id: int
    class Config:
        from_attributes = True

class FactureCreate(BaseModel):
    chantier_id: int
    montant: float

class FactureResponse(BaseModel):
    id: int
    num: str
    montant: float
    payee: bool
    chantier_id: int
    class Config:
        from_attributes = True

class RelancePayload(BaseModel):
    canal: str


# --- FONCTIONS UTILITAIRES & SÉCURITÉ ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hacher_mot_de_passe(password: str) -> str:
    return pwd_context.hash(password)

def verifier_mot_de_passe(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def creer_token_accès(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def obtenir_entreprise_courante(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
) -> EntrepriseDB:
    exception_auth = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Identifiants invalides ou token expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise exception_auth
    except jwt.PyJWTError:
        raise exception_auth

    entreprise = db.query(EntrepriseDB).filter(EntrepriseDB.email == email).first()
    if entreprise is None:
        raise exception_auth
    return entreprise


# --- APPLICATION FASTAPI ---

app = FastAPI(
    title="BatiPilot",
    description="Plateforme de gestion et de pilotage de chantiers pour le BTP"
)


# --- ROUTE TRANSCRIBE (COMPATIBILITÉ OPERA / UNIVERSEL) ---

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Traitement audio (simulé ou intégration Whisper ici)
        texte_transcrit = "Achat de matériel pour 240 euros sur le chantier"
        
        return {"text": texte_transcrit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- ROUTE AUTHENTIFICATION ---

@app.post("/auth/register", response_model=EntrepriseResponse)
def inscrire_entreprise(data: EntrepriseCreate, db: Session = Depends(get_db)):
    existant = db.query(EntrepriseDB).filter(EntrepriseDB.email == data.email).first()
    if existant:
        raise HTTPException(status_code=400, detail="Un compte existe déjà avec cet email")

    entreprise = EntrepriseDB(
        nom=data.nom,
        email=data.email,
        mot_de_passe_hache=hacher_mot_de_passe(data.mot_de_passe),
    )
    db.add(entreprise)
    db.commit()
    db.refresh(entreprise)
    return entreprise

@app.post("/auth/login", response_model=Token)
def connecter_entreprise(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    entreprise = db.query(EntrepriseDB).filter(EntrepriseDB.email == form_data.username).first()
    if not entreprise or not verifier_mot_de_passe(form_data.password, entreprise.mot_de_passe_hache):
        raise HTTPException(status_code=400, detail="Email ou mot de passe incorrect")

    token_acces = creer_token_accès(data={"sub": entreprise.email})
    return {"access_token": token_acces, "token_type": "bearer"}


# --- ROUTES SÉCURISÉES & PAGINÉES ---

@app.get("/chantiers/", response_model=List[ChantierResponse])
def lister_chantiers(
    limit: int = 10, 
    offset: int = 0, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    return (
        db.query(ChantierDB)
        .filter(ChantierDB.entreprise_id == entreprise.id)
        .offset(offset)
        .limit(limit)
        .all()
    )

@app.post("/chantiers/", response_model=ChantierResponse)
def creer_chantier(
    chantier: ChantierCreate, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    db_chantier = ChantierDB(**chantier.model_dump(), entreprise_id=entreprise.id)
    db.add(db_chantier)
    db.commit()
    db.refresh(db_chantier)
    return db_chantier

@app.get("/soustraitants/", response_model=List[SousTraitantResponse])
def lister_soustraitants(
    limit: int = 10, 
    offset: int = 0, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    return (
        db.query(SousTraitantDB)
        .filter(SousTraitantDB.entreprise_id == entreprise.id)
        .offset(offset)
        .limit(limit)
        .all()
    )

@app.post("/soustraitants/", response_model=SousTraitantResponse)
def ajouter_soustraitant(
    st: SousTraitantCreate, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    chantier = (
        db.query(ChantierDB)
        .filter(ChantierDB.id == st.chantier_id, ChantierDB.entreprise_id == entreprise.id)
        .first()
    )
    if not chantier:
        raise HTTPException(status_code=404, detail="Chantier introuvable ou non autorisé")

    db_st = SousTraitantDB(**st.model_dump(), entreprise_id=entreprise.id)
    db.add(db_st)
    chantier.cout_reel += st.montant

    db.commit()
    db.refresh(db_st)
    return db_st

@app.get("/factures/", response_model=List[FactureResponse])
def lister_factures(
    limit: int = 10, 
    offset: int = 0, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    return (
        db.query(FactureDB)
        .filter(FactureDB.entreprise_id == entreprise.id)
        .offset(offset)
        .limit(limit)
        .all()
    )

@app.post("/factures/", response_model=FactureResponse)
def creer_facture(
    f: FactureCreate, 
    db: Session = Depends(get_db),
    entreprise: EntrepriseDB = Depends(obtenir_entreprise_courante)
):
    chantier = (
        db.query(ChantierDB)
        .filter(ChantierDB.id == f.chantier_id, ChantierDB.entreprise_id == entreprise.id)
        .first()
    )
    if not chantier:
        raise HTTPException(status_code=404, detail="Chantier introuvable ou non autorisé")

    total_factures_entreprise = (
        db.query(FactureDB)
        .filter(FactureDB.entreprise_id == entreprise.id)
        .count() + 1
    )
    num_facture = f"FAC-2026-{total_factures_entreprise:03d}"

    db_facture = FactureDB(num=num_facture, entreprise_id=entreprise.id, **f.model_dump())
    db.add(db_facture)
    db.commit()
    db.refresh(db_facture)
    return db_facture

@app.post("/factures/{facture_id}/relancer")
def relancer_impaye(facture_id: int, payload: RelancePayload, db: Session = Depends(get_db)):
    facture = db.query(FactureDB).filter(FactureDB.id == facture_id).first()
    if not facture:
        raise HTTPException(status_code=404, detail="Facture introuvable")
    
    chantier = db.query(ChantierDB).filter(ChantierDB.id == facture.chantier_id).first()
    nom_chantier = chantier.nom if chantier else "Chantier"
    nom_client = chantier.client if chantier else "Client"

    texte_message = (
        f"Bonjour {nom_client},\n\n"
        f"Sauf erreur de notre part, la facture *{facture.num}* "
        f"d'un montant de *{facture.montant:,.2f} €* "
        f"concernant le chantier *{nom_chantier}* reste impayée.\n\n"
        f"Merci de procéder à son règlement dans les plus brefs délais.\n"
        f"Cordialement,\nL'équipe BatiPilot"
    )

    texte_encode = urllib.parse.quote(texte_message)
    telephone_client = "33600000000" 
    whatsapp_url = f"https://wa.me/{telephone_client}?text={texte_encode}"

    return {
        "statut": "succès",
        "canal": payload.canal,
        "facture": facture.num,
        "whatsapp_url": whatsapp_url if payload.canal == "whatsapp" else None,
        "message_preview": texte_message
    }