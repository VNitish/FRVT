# auth/jwt_utils.py
from jose import jwt, JWTError
from datetime import datetime, timedelta
from frt.commons.config import Config
from passlib.context import CryptContext

secret_key = Config.JWT_SECRET_KEY
algorithm = Config.JWT_ALGORITHM
access_token_exp_mins = Config.ACCESS_TOKEN_EXPIRE_MINUTES

with open(Config.JWT_PRIVATE_KEY_PATH, "r") as f:
    private_key = f.read()

with open(Config.JWT_PUBLIC_KEY_PATH, "r") as f:
    public_key = f.read()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: int = 30):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, private_key, algorithm=algorithm), expire


def verify_access_token(token: str):
    try:
        payload = jwt.decode(token, public_key, algorithms=["RS256"])
        return payload
    except JWTError:
        return None
