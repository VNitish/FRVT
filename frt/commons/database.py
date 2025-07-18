from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(".env")

class Dbconnect:

    def __init__(self, db_name, db_url):
        self.url = db_url
        self.db_name = db_name
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]