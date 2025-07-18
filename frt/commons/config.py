import os


class Config:
    MONGO_DB_URL = os.getenv('MONGO_DB_URL')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
    JWT_SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_ALGORITHM = os.getenv('ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 60))
    JWT_PUBLIC_KEY_PATH = './Utilities/public_key.pem'
    JWT_PRIVATE_KEY_PATH = './Utilities/private_key.pem'
