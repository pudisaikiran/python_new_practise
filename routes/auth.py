from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from database import get_db, User, RefreshToken
from security import hash_password, verify_password, create_access_token, hash_refresh_token, generate_refresh_token

router = APIRouter(prefix="/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    email: str
    full_name: str

class LogoutRequest(BaseModel):
    refresh_token: str

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == body.email).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")
        
    new_user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        is_active=True
    )
    db.add(new_user)
    db.commit()
    return {"message": "Account created successfully. You can now log in."}

@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Incorrect email or password"
    )
    
    if not user or not verify_password(body.password, user.hashed_password):
        raise invalid
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")
    
    # Generate tokens
    access_token = create_access_token(data={"sub": user.email})
    raw_refresh_token = generate_refresh_token()
    
    # Store refresh token in database (hashed for security)
    new_refresh_token = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh_token),
        expires_at=datetime.utcnow() + timedelta(days=7),
        created_at=datetime.utcnow(),
        revoked=False
    )
    db.add(new_refresh_token)
    db.commit()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh_token,
        token_type="bearer",
        email=user.email,
        full_name=user.full_name
    )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: LogoutRequest, db: Session = Depends(get_db)):
    token_hash = hash_refresh_token(body.refresh_token)
    stored = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    
    if stored is not None:
        # Physically delete ("kill") the refresh token from the database
        db.delete(stored)
        db.commit()
