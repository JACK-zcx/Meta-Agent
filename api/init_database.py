from sqlmodel import create_engine,Session
import os

# Set the path for the database file (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(BASE_DIR, 'db')
os.makedirs(db_dir, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(db_dir, 'database.db')}"

# Create the database engine
engine = create_engine(DATABASE_URL,echo=False)