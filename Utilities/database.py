from frt.commons.database import Dbconnect
from frt.commons.config import Config

DB_NAME = Config.MONGO_DB_NAME
MONGODB_CON_STR  = Config.MONGO_DB_URL
db_obj = Dbconnect(db_name=DB_NAME, db_url=MONGODB_CON_STR)
db = db_obj.db