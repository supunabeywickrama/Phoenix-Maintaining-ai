import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unified_rag.db.database import engine, Base
import unified_rag.db.models # Import models to register them with Base

def init_db():
    print("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
