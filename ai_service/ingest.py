# ai_service/ingest.py
import os
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.pgvector import PGVectorStore
from vanna.base import VannaBase

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

class MyVanna(VannaBase):
    pass

# Setup LLM + Vector DB
llm = OpenAI_Chat(api_key=OPENAI_API_KEY)
vectorstore = PGVectorStore(connection_string=DB_URL, dimension=1536)
vn = MyVanna(llm=llm, vectorstore=vectorstore)

# Ingest schema
def ingest_schema():
    schema_text = """
    Table: members
      - member_id (PK)
      - name
      - date_of_birth
      - insurance_id (FK)

    Table: claims
      - claim_id (PK)
      - member_id (FK)
      - amount
      - status
    """
    vn.train(document=schema_text, doc_type="schema")

# Ingest training examples
def ingest_training_data():
    examples = [
        {"question": "List all members", "sql": "SELECT * FROM members;"},
        {"question": "How many claims are pending?", "sql": "SELECT COUNT(*) FROM claims WHERE status='Pending';"}
    ]
    for ex in examples:
        vn.train(document=ex, doc_type="training_data")

if __name__ == "__main__":
    ingest_schema()
    ingest_training_data()
    print("✅ Ingestion complete, knowledge stored in Postgres pgvector")
