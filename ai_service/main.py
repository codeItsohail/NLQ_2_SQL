# === add imports near top ===
import os, time, uuid, re
from fastapi import HTTPException
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from datetime import datetime
from fastapi import BackgroundTasks
import pandas as pd
from fastapi.responses import FileResponse
from fastapi import Response
import json
from sqlalchemy import create_engine, text
import urllib.parse
from uuid import uuid4
import urllib.parse
# pyright: reportMissingImports=false
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from urllib.parse import quote_plus

import os, time, uuid, re
from typing import List, Tuple
from urllib.parse import urlparse
import psycopg2

# Optional DB drivers
try:
    import mysql.connector as mysql_connector
except Exception:
    mysql_connector = None

try:
    import pyodbc as pyodbc_driver
except Exception:
    pyodbc_driver = None

# SQL Server connection string builder (moved from utils)
def build_sqlserver_connection_string(
    host: str, 
    port: Optional[int], 
    database: str, 
    username: str, 
    password: str, 
    use_sqlalchemy: bool = False
) -> str:
    """
    Build SQL Server connection string for both pyodbc and SQLAlchemy formats.
    """
    port = port or 1433
    
    if use_sqlalchemy:
        # SQLAlchemy format for Vanna
        encoded_username = urllib.parse.quote(username, safe="")
        encoded_password = urllib.parse.quote_plus(password)
        
        conn_str = (
            f"mssql+pyodbc://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
            f"?driver=ODBC+Driver+17+for+SQL+Server"
            f"&Encrypt=yes"
            f"&TrustServerCertificate=yes"
            f"&Connection+Timeout=30"
        )
        return conn_str
    else:
        # Direct pyodbc connection string
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={host},{port};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=30;"
        )
        return conn_str

# Get default schema based on database type
def get_default_schema(db_type: str) -> Optional[str]:
    """Return the default schema name for the database type.
    For MySQL we return None because MySQL uses the database (catalog) instead of schemas.
    """
    db_type = (db_type or "").lower()
    if db_type in ("sqlserver", "mssql"):
        return "dbo"
    elif db_type in ("postgresql", "postgres"):
        return "public"
    elif db_type == "mysql":
        return None
    return None

def get_current_database_name(db_url: str) -> Optional[str]:
    p = parse_db_url(db_url)
    return p.get("db")


# --- Vanna with OpenAI + ChromaDB (local vector store) ---
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        if config is None:
            config = {}
        config['path'] = config.get('path', './chroma_db')
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        self.system_prompt = None
        
    def set_system_prompt(self, content: str):
        self.system_prompt = content

    def generate_sql(self, question: str):
        if self.system_prompt:
            question = f"{self.system_prompt}\n\nUser question: {question}"
        return super().generate_sql(question , allow_llm_to_see_data = True)

    def connect_to_database(self, connection_string: str):
        """Method to connect Vanna to a database."""
        try:
            # Assuming `self.db` is where the database connection is stored
            self.db = connect_target(connection_string)
            print("Connected to the database.")
        except Exception as e:
            raise ValueError(f"Failed to connect to database: {e}")



load_dotenv()

app = FastAPI()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
APP_DB_URL = os.getenv("APP_DATABASE_URL")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma-persist")

if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)

# --- Request body schema ---
class ConnectionData(BaseModel):
    db_type: str  # "postgresql", "mysql", "sqlserver"
    host: str
    db_name: str
    user: str
    password: str
    port: int

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Removed the table creation startup event since you want to remove it

# --- Enhanced SQL Server connection handling ---
@app.post("/test-connection")
def test_connection(data: ConnectionData):
    """Test database connectivity with enhanced SQL Server support."""
    try:
        if data.db_type.lower() == "postgresql":
            conn = psycopg2.connect(
                host=data.host,
                dbname=data.db_name,
                user=data.user,
                password=data.password,
                port=data.port,
                connect_timeout=10
            )
            conn.close()
            return {"success": True, "message": "PostgreSQL connection successful"}

        elif data.db_type.lower() == "mysql":
            if mysql_connector is None:
                return {"success": False, "error": "MySQL connector not available"}
            
            conn = mysql_connector.connect(
                host=data.host,
                database=data.db_name,
                user=data.user,
                password=data.password,
                port=data.port,
                connection_timeout=10
            )
            conn.close()
            return {"success": True, "message": "MySQL connection successful"}

        elif data.db_type.lower() == "sqlserver":
            if pyodbc_driver is None:
                return {"success": False, "error": "pyodbc not available"}
                
            # Test with pyodbc first
            conn_str = build_sqlserver_connection_string(
                data.host, data.port, data.db_name, data.user, data.password, use_sqlalchemy=False
            )
            conn = pyodbc_driver.connect(conn_str, timeout=10)
            conn.close()
            
            # Also test SQLAlchemy format for Vanna compatibility
            sqlalchemy_str = build_sqlserver_connection_string(
                data.host, data.port, data.db_name, data.user, data.password, use_sqlalchemy=True
            )
            engine = create_engine(sqlalchemy_str)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return {"success": True, "message": "SQL Server connection successful (both pyodbc and SQLAlchemy)"}

        else:
            return {"success": False, "error": f"Unsupported database type: {data.db_type}"}

    except Exception as e:
        error_msg = str(e)
        if "ODBC Driver" in error_msg:
            error_msg += "\nTip: Install ODBC Driver 17 for SQL Server"
        return {"success": False, "error": error_msg}

# ---------- helpers ----------

def build_db_url(dbtype: str, host: str, port: int, dbname: str, user: str, password: str) -> str:
    dbtype = dbtype.lower()
    if dbtype == "sqlserver":
        # SQLAlchemy + pyodbc URL
        return (
            f"mssql+pyodbc://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{dbname}?driver=ODBC+Driver+17+for+SQL+Server"
        )
    elif dbtype in ["postgres", "postgresql"]:
        return (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{dbname}"
        )
    elif dbtype == "mysql":
        return (
            f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{dbname}"
        )
    else:
        raise ValueError(f"Unsupported dbtype: {dbtype}")


def _normalize_pg_url(url: str) -> str:
    return url.replace("postgres://", "postgresql://")

def appdb():
    """Connect to MySQL app database"""
    if not APP_DB_URL:
        raise RuntimeError("APP_DATABASE_URL not set")
    
    # Parse MySQL connection string
    # Handle both mysql:// and mysql+pymysql:// formats
    url = APP_DB_URL.replace("mysql+pymysql://", "mysql://").replace("mysql+mysqlconnector://", "mysql://")
    parsed = urlparse(url)
    
    # URL decode password if it contains encoded characters
    password = urllib.parse.unquote(parsed.password) if parsed.password else None
    
    # Parse query parameters for SSL and other options
    query_params = {}
    if parsed.query:
        from urllib.parse import parse_qs
        query_params = parse_qs(parsed.query)
    
    # Base connection config
    conn_config = {
        'host': parsed.hostname,
        'port': parsed.port or 3306,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': password,
        'autocommit': False,
        'connect_timeout': 10
    }
    
    # Add SSL configuration for Azure MySQL if needed
    # Azure MySQL typically requires TLS
    if parsed.hostname and 'mysql.database.azure.com' in parsed.hostname:
        # Ensure TLS is enabled
        conn_config['ssl_disabled'] = False
        # If caller provided a CA bundle, use it; otherwise relax verification to avoid handshake failures
        if 'ssl_ca' in query_params and query_params['ssl_ca']:
            conn_config['ssl_ca'] = query_params['ssl_ca'][0]
            # When a CA is supplied, you may enable verification explicitly
            conn_config['ssl_verify_cert'] = True
            conn_config['ssl_verify_identity'] = False
        else:
            # No CA provided; require TLS but disable certificate verification to prevent SSL handshake errors
            conn_config['ssl_verify_cert'] = False
            conn_config['ssl_verify_identity'] = False

    # First attempt
    try:
        return mysql_connector.connect(**conn_config)
    except Exception as e:
        # Fallback for SSL/TLS handshake issues (Azure MySQL)
        fallback = dict(conn_config)
        fallback['ssl_disabled'] = False
        fallback['ssl_verify_cert'] = False
        fallback['ssl_verify_identity'] = False
        # Prefer TLS 1.2+ explicitly
        try:
            fallback['tls_versions'] = ['TLSv1.3', 'TLSv1.2']
        except Exception:
            pass
        # Some environments need pure Python implementation for TLS/SNI handling
        fallback['use_pure'] = True
        return mysql_connector.connect(**fallback)

def set_vanna_collection(v, agent_id: str):
    try:
        v.set_collection_name(agent_id)
    except Exception:
        v.config = {**getattr(v, "config", {}), "collection_name": agent_id}
    if hasattr(v, "set_persist_directory"):
        v.set_persist_directory(CHROMA_DIR)
    else:
        v.config = {**getattr(v, "config", {}), "persist_directory": CHROMA_DIR}

def parse_db_url(db_url: str):
    """Enhanced URL parsing with SQL Server support"""
    if db_url.startswith("sqlserver://"):
        # Handle custom sqlserver:// format
        u = urlparse(db_url.replace("sqlserver://", "mssql://"))
        return dict(
            scheme="sqlserver", host=u.hostname, port=u.port or 1433,
            user=u.username, password=u.password, db=u.path.lstrip("/")
        )
    elif db_url.startswith("mssql"):
        u = urlparse(db_url)
        return dict(
            scheme="sqlserver", host=u.hostname, port=u.port or 1433,
            user=u.username, password=u.password, db=u.path.lstrip("/")
        )
    else:
        u = urlparse(db_url.replace("postgres://", "postgresql://"))
        return dict(
            scheme=u.scheme, host=u.hostname, port=u.port,
            user=u.username, password=u.password, db=u.path.lstrip("/")
        )

def connect_target(db_url: str):
    """Enhanced connection with SQL Server support"""
    p = parse_db_url(db_url)
    if p["scheme"] == "postgresql":
        return psycopg2.connect(host=p["host"], port=p["port"], dbname=p["db"],
                                user=p["user"], password=p["password"], connect_timeout=10)
    elif p["scheme"] == "mysql":
        if mysql_connector is None:
            raise RuntimeError("MySQL connector not available")
        return mysql_connector.connect(host=p["host"], port=p["port"], database=p["db"],
                                       user=p["user"], password=p["password"], connection_timeout=10)
    elif p["scheme"] == "sqlserver":
        if pyodbc_driver is None:
            raise RuntimeError("pyodbc not available")
        conn_str = build_sqlserver_connection_string(
            p["host"], p["port"], p["db"], p["user"], p["password"], use_sqlalchemy=False
        )
        return pyodbc_driver.connect(conn_str, timeout=10)
    raise ValueError(f"Unsupported scheme: {p['scheme']}")

def get_vanna_connection_string(db_url: str) -> str:
    """Get SQLAlchemy-compatible connection string for Vanna"""
    p = parse_db_url(db_url)
    if p["scheme"] == "sqlserver":
        return build_sqlserver_connection_string(
            p["host"], p["port"], p["db"], p["user"], p["password"], use_sqlalchemy=True
        )
    return db_url.replace("postgres://", "postgresql://")

def get_agent(conn, agent_id: str):
    """Get agent from MySQL database"""
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT agent_id, user_id, agent_name, agent_description, 
               create_date, update_date, is_active
        FROM agent 
        WHERE agent_id=%s AND is_active=1
    """, (agent_id,))
    r = cur.fetchone()
    cur.close()
    
    if not r:
        return None
    
    # Get db_config for this agent
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT db_type, db_connection_string, db_host, db_name, 
               db_user, db_pass, db_port
        FROM db_configs 
        WHERE agent_id=%s AND is_active=1
        LIMIT 1
    """, (agent_id,))
    db_config = cur.fetchone()
    cur.close()
    
    db_url = None
    db_type = None
    if db_config:
        if db_config['db_connection_string']:
            db_url = db_config['db_connection_string']
        else:
            # Build connection string from components
            db_type = db_config['db_type']
            db_url = build_db_url(
                db_config['db_type'],
                db_config['db_host'],
                db_config['db_port'],
                db_config['db_name'],
                db_config['db_user'],
                db_config['db_pass']
            )
        db_type = db_config['db_type']
    
    return {
        "id": r['agent_id'],
        "user_id": r['user_id'],
        "name": r['agent_name'],
        "description": r['agent_description'],
        "db_url": db_url,
        "trained_at": r['update_date'],  # Using update_date as trained_at
        "dbtype": db_type
    }

def info_schema_text(db_url: str, db_type: str = None) -> str:
    """Enhanced schema extraction with proper schema detection"""
    p = parse_db_url(db_url)
    default_schema = get_default_schema(db_type or p["scheme"])
    
    with connect_target(db_url) as conn:
        cur = conn.cursor()
        if p["scheme"] == "postgresql":
            cur.execute("""
              SELECT table_schema, table_name, column_name, data_type
              FROM information_schema.columns
              WHERE table_schema = %s
              ORDER BY table_name, ordinal_position
            """, (default_schema,))
        elif p["scheme"] == "mysql":
            cur.execute("""
              SELECT table_schema, table_name, column_name, data_type
              FROM information_schema.columns
              WHERE table_schema = DATABASE()
              ORDER BY table_name, ordinal_position
            """)
        elif p["scheme"] == "sqlserver":
            cur.execute("""
              SELECT 
                SCHEMA_NAME(t.schema_id) as table_schema,
                t.name as table_name,
                c.name as column_name,
                ty.name as data_type
              FROM sys.tables t
              INNER JOIN sys.columns c ON t.object_id = c.object_id
              INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
              WHERE SCHEMA_NAME(t.schema_id) = %s
              ORDER BY t.name, c.column_id
            """, (default_schema,))
        rows = cur.fetchall()
    return "\n".join(f"{r[0]}.{r[1]}.{r[2]} {r[3]}" for r in rows)

def run_sql_preview(db_url: str, sql: str, limit: int = 50):
    """Enhanced SQL execution with SQL Server support"""
    s = sql.strip().rstrip(";")
    low = s.lower()
    p = parse_db_url(db_url)
    
    if "limit " not in low and " top " not in low:
        if p["scheme"] in ("postgresql", "mysql"):
            s = f"{s} LIMIT {limit}"
        elif p["scheme"] == "sqlserver" and low.startswith("select "):
            s = f"SELECT TOP {limit} " + s[7:]
    
    with connect_target(db_url) as conn:
        cur = conn.cursor()
        cur.execute(s)
        cols = [c[0] for c in (cur.description or [])]
        rows = cur.fetchall() if cur.description else []
        data = [dict(zip(cols, r)) for r in rows] if cols else []
    return cols, data

def _mask_url(url: str) -> str:
    # mask the password portion between ":" and "@" once
    if ":" in url and "@" in url:
        before, after = url.split("@", 1)
        if ":" in before:
            head, _pwd = before.rsplit(":", 1)
            return f"{head}:****@{after}"
    return url

def apply_system_prompts(v, conn, agent_id: str):
    """Apply system prompts from filters table"""
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT nlq 
        FROM filters 
        WHERE agent_id=%s AND is_active=1 AND nlq IS NOT NULL
    """, (agent_id,))
    rows = cur.fetchall()
    cur.close()
    
    for row in rows:
        content = row['nlq']
        if content and content.strip():
            try:
                v.train(question="SYSTEM_PROMPT", sql=None, ddl=None, documentation=content)
            except Exception as e:
                print(f"Failed to apply system prompt: {e}")

# ---------- request models ----------
class AskBody(BaseModel):
    question: str
    agent_id: str
    execute: bool = True
    user_id: Optional[str] = None
    limit: int = 50
    session_id: Optional[str] = None

class FeedbackBody(BaseModel):
    question: str
    sql: str
    valid: bool
    agent_id: str
    user_id: Optional[str] = None
    answer: Optional[str] = None
    corrected_sql: Optional[str] = None

class ExecuteBody(BaseModel):
    agent_id: str
    sql: str
    limit: int = 50
    user_id: Optional[str] = None

class TrainStartBody(BaseModel):
    agent_id: str

class CreateAgentBody(BaseModel):
    agent_id: Optional[str] = None
    user_id: str
    name: str
    description: Optional[str] = None
    
    # Either pass full db_url OR the parts below (we'll build one)
    db_url: Optional[str] = None
    host: Optional[str] = None
    dbName: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    port: Optional[int] = None
    dbType: Optional[str] = None  # "sqlserver" | "mysql" | "postgres"
class SavePromptBody(BaseModel):
    agent_id: str
    user_id: str
    question: str
    sql_query: str
    answer: Optional[str] = None

# ---------- Vanna/OpenAI helpers ----------
_VN_SINGLETON = None

def get_vanna():
    """Return a singleton Vanna instance configured for OpenAI + ChromaDB (local)."""
    global _VN_SINGLETON
    if _VN_SINGLETON is not None:
        return _VN_SINGLETON

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment")
        return None
        
    api_type = os.getenv("OPENAI_API_TYPE", "").strip().lower()
    cfg = {}

    if api_type == "azure" or os.getenv("AZURE_OPENAI_API_BASE"):
        cfg = {
            "api_type": "azure",
            "api_base": os.getenv("AZURE_OPENAI_API_BASE", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            "engine": os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_ENGINE", "")),
            "api_key": api_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "allow_llm_to_see_data" : True,
        }
    else:
        cfg = {
            "api_key": api_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "allow_llm_to_see_data" : True,
        }

    try:
        _VN_SINGLETON = MyVanna(config=cfg)
    except Exception as e:
        print(f"Vanna initialization error: {e}")
        _VN_SINGLETON = None
    return _VN_SINGLETON

# ---------- TRAIN: start ----------
@app.post("/train/start")
def train_start(body: TrainStartBody, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    try:
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"error": "Agent not found", "run_id": None, "status": "failed"}
            
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO training_runs (training_runs_id, agent_id, status, progress, message, 
                                          create_date, started_at)
                VALUES (%s, %s, 'queued', 0, 'Queued', NOW(), NOW())
            """, (run_id, body.agent_id))
            conn.commit()
            cur.close()
            
        background_tasks.add_task(_train_job, run_id, body.agent_id)
        return {"run_id": run_id, "status": "queued", "progress": 0}
    except Exception as e:
        print(f"Error starting training: {e}")
        return {"error": str(e), "run_id": None, "status": "failed"}

def _train_job(run_id: str, agent_id: str):
    def upd(conn, prog, status, msg):
        cur = conn.cursor()
        cur.execute("""
            UPDATE training_runs 
            SET progress=%s, status=%s, message=%s, update_date=NOW()
            WHERE training_runs_id=%s
        """, (prog, status, msg, run_id))
        cur.execute("""
            INSERT INTO training_logs(training_runs_id, ts, progress, message) 
            VALUES (%s, NOW(), %s, %s)
        """, (run_id, prog, msg))
        conn.commit()
        cur.close()

    try:
        with appdb() as conn:
            upd(conn, 5, "running", "Loading agent")
            agent = get_agent(conn, agent_id)
            if not agent:
                upd(conn, 100, "failed", "Agent not found")
                return
            db_url = agent["db_url"]
            db_type = agent.get("dbtype", "postgresql")
            
            upd(conn, 15, "running", f"Reading database schema ({db_type})")
            try:
                schema_text = info_schema_text(db_url, db_type)
                if not schema_text.strip():
                    upd(conn, 100, "failed", "No schema found in database")
                    return
            except Exception as e:
                upd(conn, 100, "failed", f"Failed to read schema: {str(e)}")
                return

            upd(conn, 35, "running", "Initializing AI training")
            v = get_vanna()
            if not v:
                upd(conn, 100, "failed", "Vanna not available")
                return
            
            set_vanna_collection(v, agent_id)

            # Apply system prompts (if any)
            with appdb() as conn2:
                apply_system_prompts(v, conn2, agent_id)
            
            upd(conn, 45, "running", "Connecting Vanna to database")
            try:
                vanna_conn_str = get_vanna_connection_string(db_url)
                v.connect_to_database(vanna_conn_str)
            except Exception as e:
                upd(conn, 100, "failed", f"Failed to connect: {e}")
                return
            
            upd(conn, 55, "running", "Training on database schema")
            try:
                default_schema = get_default_schema(db_type)    
                if (db_type or "").lower() == "mysql":
                    current_db = get_current_database_name(db_url) or "CURRENT_DB"
                    schema_prompt = (
                        "For MySQL, do NOT prefix tables with a schema; use bare table names. "
                        f"If qualification is absolutely required, qualify with `{current_db}`."
                    )
                else:
                    schema_prompt = f"Always use schema '{default_schema}' unless another schema is explicitly specified."
                v.train(ddl=schema_text + "\n\n" + schema_prompt)
            except Exception as e:
                upd(conn, 100, "failed", f"Failed to train: {e}")
                return

            upd(conn, 75, "running", "Loading validated Q&A pairs")
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT favorite_name as question, favorite_query as sql_query 
                FROM favorite_queries 
                WHERE agent_id=%s AND is_active=1
                ORDER BY favorite_query_id
            """, (agent_id,))
            pairs = cur.fetchall()
            cur.close()
                
            total = max(1, len(pairs))
            for i, pair in enumerate(pairs, start=1):
                try:
                    v.train(question=pair['question'], sql=pair['sql_query'])
                    upd(conn, 75 + int(20*i/total), "running", f"Training Q&A {i}/{len(pairs)}")
                except Exception as e:
                    print(f"Error training Q&A pair {i}: {e}")

            cur = conn.cursor()
            cur.execute("""
                UPDATE agent SET update_date=NOW(), update_by=1 
                WHERE agent_id=%s
            """, (agent_id,))
            conn.commit()
            cur.close()
            
            cur = conn.cursor()
            cur.execute("""
                UPDATE training_runs 
                SET finished_at=NOW() 
                WHERE training_runs_id=%s
            """, (run_id,))
            conn.commit()
            cur.close()
            
            upd(conn, 100, "succeeded", "Training complete")
            
    except Exception as e:
        print(f"Training job error: {e}")
        with appdb() as conn:
            upd(conn, 100, "failed", f"Training failed: {e}")

@app.get("/train/status/{run_id}")
def train_status(run_id: str):
    try:
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT agent_id, status, progress, message, started_at, finished_at 
                FROM training_runs 
                WHERE training_runs_id=%s
            """, (run_id,))
            r = cur.fetchone()
            cur.close()
            
            if not r:
                return {"error": "not_found"}
            return {
                "run_id": run_id,
                "agent_id": r['agent_id'],
                "status": r['status'],
                "progress": r['progress'],
                "message": r['message'],
                "started_at": r['started_at'],
                "finished_at": r['finished_at']
            }
    except Exception as e:
        return {"error": str(e)}

# ---------- ASK (generate SQL -> safe execute) ----------
@app.post("/ask")
def ask(body: AskBody):
    t0 = time.time()
    session_id = body.session_id or str(uuid.uuid4())
    
    try:
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"answer":"Agent not found","sql":"","data":{"columns":[],"rows":[]},"session_id":session_id}
            db_url = agent["db_url"]
            db_type = agent.get("dbtype", "postgresql")

        v = get_vanna()
        if not v:
            error_msg = "Vanna not available"
            return {"answer": f"Failed: {error_msg}", "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        set_vanna_collection(v, body.agent_id)

        # Apply schema-specific system prompt
        default_schema = get_default_schema(db_type)
        
        dtype = (db_type or "").lower()
        if dtype == "mysql":
            current_db = get_current_database_name(db_url) or "CURRENT_DB"
            schema_instruction = (
                "For MySQL, do NOT prefix tables with a schema; use bare table names. "
                f"If qualification is required, qualify with `{current_db}`.\n"
                "If asked for the current database name, use: SELECT DATABASE();"
            )
        elif dtype in ("sqlserver", "mssql"):
            schema_instruction = (
                f"For SQL Server, use schema '{default_schema or 'dbo'}' unless specified. "
                "If asked for the current database name, use: SELECT DB_NAME();"
            )
        elif dtype in ("postgres", "postgresql"):
            schema_instruction = (
                f"For PostgreSQL, use schema '{default_schema or 'public'}' unless specified. "
                "If asked for the current database name, use: SELECT current_database();"
            )
        else:
            schema_instruction = (
                f"Use schema '{default_schema or 'public'}' unless specified."
            )

        all_prompts = [schema_instruction]
        with appdb() as conn2:
            cur = conn2.cursor(dictionary=True)
            cur.execute("""
                SELECT nlq 
                FROM filters 
                WHERE agent_id=%s AND is_active=1 AND nlq IS NOT NULL
            """, (body.agent_id,))
            for row in cur.fetchall():
                content = row['nlq']
                if content and content.strip():
                    all_prompts.append(content.strip())
            cur.close()

        v.set_system_prompt("\n\n".join(all_prompts))

        try:
            vanna_conn_str = get_vanna_connection_string(db_url)
            v.connect_to_database(vanna_conn_str)
        except Exception as e:
            error_msg = f"Failed to connect: {e}"
            return {"answer": error_msg, "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        # Generate SQL
        sql = ""
        try:
            sql = v.generate_sql(body.question)
        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            return {"answer": error_msg, "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        columns, rows, err = [], [], None
        if body.execute and sql.strip():
            try:
                columns, rows = run_sql_preview(db_url, sql, body.limit)
            except Exception as ex:
                err = str(ex)

        dur_ms = int((time.time()-t0)*1000)
        answer = "Query executed successfully." if not err else f"Execution error: {err}"

        return {"answer": answer, "sql": sql, "data": {"columns": columns, "rows": rows}, "session_id": session_id}
        
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return {"answer": f"Internal error: {str(e)}", "sql": "", "data": {"columns": [], "rows": []}, "session_id": session_id}

# ---------- EXECUTE (manual run) ----------
@app.post("/execute")
def execute_query(body: ExecuteBody):
    """
    Execute raw SQL against the agent's configured target DB with a preview limit.
    Returns columns and rows on success, or an error string on failure.
    """
    try:
        # Resolve agent -> connection string
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}
            db_url = agent["db_url"]

        # Run with preview/limit handling per RDBMS
        limit = body.limit or 50
        cols, data = run_sql_preview(db_url, body.sql, limit)
        return {
            "success": True,
            "columns": cols,
            "rows": data,
            "sql": body.sql
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# ---------- EXECUTE (manual run) ----------
@app.post("/execute")
def execute_query(body: ExecuteBody):
    """
    Execute raw SQL against the agent's configured target DB with a preview limit.
    Returns columns and rows on success, or an error string on failure.
    """
    try:
        # Resolve agent -> connection string
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}
            db_url = agent["db_url"]

        # Run with preview/limit handling per RDBMS
        limit = body.limit or 50
        cols, data = run_sql_preview(db_url, body.sql, limit)
        return {
            "success": True,
            "columns": cols,
            "rows": data,
            "sql": body.sql
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# ---------- FEEDBACK with corrected SQL support ----------
@app.post("/feedback")
def feedback(body: FeedbackBody):
    try:
        with appdb() as conn:
            final_sql = body.corrected_sql if body.corrected_sql else body.sql
            
            cur = conn.cursor()
            # Store in favorite_queries if valid
            if body.valid:
                cur.execute("""
                    INSERT INTO favorite_queries (agent_id, favorite_name, favorite_query, 
                                                 description, created_by)
                    VALUES (%s, %s, %s, %s, %s)
                """, (body.agent_id, body.question[:150], final_sql, 
                      body.answer or "Validated query", body.user_id or 1))
            conn.commit()
            cur.close()
            
        return {"success": True, "message": "Feedback saved successfully"}
    except Exception as e:
        print(f"Error in feedback endpoint: {e}")
        return {"success": False, "error": str(e)}

# ---------- Save prompt endpoint ----------
@app.post("/prompts/save")
def save_prompt(body: SavePromptBody):
    """Save a prompt to favorite_queries for future reference"""
    try:
        with appdb() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO favorite_queries (agent_id, favorite_name, favorite_query, 
                                             description, created_by, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (body.agent_id, body.question[:150], body.sql_query, 
                  body.answer or "Saved prompt", body.user_id or 1))
            conn.commit()
            prompt_id = cur.lastrowid
            cur.close()
            
        return {"success": True, "prompt_id": prompt_id, "message": "Prompt saved successfully"}
    except Exception as e:
        print(f"Error saving prompt: {e}")
        return {"success": False, "error": str(e)}

# ---------- Get saved prompts ----------
@app.get("/prompts/saved/{agent_id}")
def get_saved_prompts(agent_id: str, user_id: str = None):
    """Get all saved prompts for an agent"""
    try:
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            if user_id:
                cur.execute("""
                    SELECT favorite_query_id as id, favorite_name as question, 
                           favorite_query as sql_query, description as answer, 
                           created_at
                    FROM favorite_queries 
                    WHERE agent_id=%s AND created_by=%s AND is_active=1
                    ORDER BY created_at DESC
                """, (agent_id, user_id))
            else:
                cur.execute("""
                    SELECT favorite_query_id as id, favorite_name as question, 
                           favorite_query as sql_query, description as answer, 
                           created_at
                    FROM favorite_queries 
                    WHERE agent_id=%s AND is_active=1
                    ORDER BY created_at DESC
                """, (agent_id,))
            rows = cur.fetchall()
            cur.close()
        
        prompts = [
            {
                "id": r['id'],
                "question": r['question'],
                "sql_query": r['sql_query'],
                "answer": r['answer'],
                "created_at": r['created_at'].isoformat() if r['created_at'] else None
            }
            for r in rows
        ]
        return {"success": True, "prompts": prompts}
    except Exception as e:
        print(f"Error getting saved prompts: {e}")
        return {"success": False, "error": str(e), "prompts": []}

# ---------- Favourites: list + detail ----------
@app.get("/favorites/{agent_id}")
def list_favorites(agent_id: str):
    """List saved favourite queries for an agent (id, name, created_at)."""
    try:
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT
                    favorite_query_id as id,
                    favorite_name as name,
                    created_at
                FROM favorite_queries
                WHERE agent_id=%s AND is_active=1
                ORDER BY created_at DESC
            """, (agent_id,))
            rows = cur.fetchall()
            cur.close()
        return {"success": True, "favorites": rows}
    except Exception as e:
        print(f"Error list_favorites: {e}")
        return {"success": False, "error": str(e), "favorites": []}

@app.get("/favorites/detail/{favorite_id}")
def favorite_detail(favorite_id: int):
    """Get favourite query details by id.
    Tries stored procedure sp_get_favorite_query_by_id(favorite_id)
    and falls back to direct SELECT if SP is absent.
    Returns: { id, favorite_name, favorite_query, description, created_at }
    Also maps description (NLQ) so the UI can prefill NLQ and SQL editors.
    """
    try:
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            # Try stored procedure first
            tried_proc = False
            try:
                tried_proc = True
                cur.callproc("sp_get_favorite_query_by_id", [favorite_id])
                # Some drivers require fetching from next result set
                # Fetch first available result set rows
                result_rows = []
                # Try to iterate over available result sets
                for _ in range(2):  # safe small loop
                    rows = cur.fetchall()
                    if rows:
                        result_rows = rows
                        break
                    # Advance if supported
                    try:
                        cur.nextset()
                    except Exception:
                        break
                if not result_rows:
                    # If no rows from SP, fall back to select
                    raise Exception("SP returned no rows")
                row = result_rows[0]
            except Exception as _:
                # Fallback to direct select
                cur.close()
                cur = conn.cursor(dictionary=True)
                cur.execute("""
                    SELECT
                        favorite_query_id as id,
                        favorite_name,
                        favorite_query,
                        description,
                        created_at
                    FROM favorite_queries
                    WHERE favorite_query_id=%s AND is_active=1
                    LIMIT 1
                """, (favorite_id,))
                row = cur.fetchone() or {}

            cur.close()

        if not row:
            return {"success": False, "error": "not_found"}

        # Normalize keys for frontend use
        return {
            "success": True,
            "favorite": {
                "id": row.get("id") or row.get("favorite_query_id"),
                "favorite_name": row.get("favorite_name") or row.get("name"),
                "favorite_query": row.get("favorite_query") or row.get("sql_query"),
                "description": row.get("description") or row.get("nlq") or "",
                "created_at": (row.get("created_at").isoformat()
                               if hasattr(row.get("created_at"), "isoformat") else row.get("created_at"))
            }
        }
    except Exception as e:
        print(f"Error favorite_detail: {e}")
        return {"success": False, "error": str(e)}

# ---------- Create Agents ----------
@app.post("/agents")
def create_agent(b: CreateAgentBody):
    try:
        if not b.user_id or not b.name:
            raise HTTPException(status_code=400, detail="user_id and name are required")
        if len(b.name) > 100:
            raise HTTPException(status_code=400, detail="Agent name must be less than 100 characters")

        uid = int(str(b.user_id)) if str(b.user_id).isdigit() else 1
        final_db_url = b.db_url
        final_port = b.port
        final_dbtype = (b.dbType or "").lower() if b.dbType else None

        if not final_db_url:
            if not (final_dbtype and b.host and b.dbName and b.user is not None and b.password is not None):
                raise HTTPException(
                    status_code=400,
                    detail="When db_url is not provided, dbType, host, dbName, user, password are required"
                )
            final_db_url = build_db_url(final_dbtype, b.host, b.port or 5432, b.dbName, b.user, b.password)

            if final_port is None:
                if final_dbtype == "sqlserver":
                    final_port = 1433
                elif final_dbtype == "mysql":
                    final_port = 3306
                else:
                    final_port = 5432
        else:
            # If full URL is provided but dbType/port missing, best-effort infer
            if not final_dbtype:
                if final_db_url.startswith("mssql+"):
                    final_dbtype = "sqlserver"
                elif final_db_url.startswith("mysql+"):
                    final_dbtype = "mysql"
                elif final_db_url.startswith("postgresql+"):
                    final_dbtype = "postgres"
            # port extraction is optional; we store provided port if any
            if final_port is None:
                # try to parse ":port/" pattern
                try:
                    after_at = final_db_url.split("@", 1)[1]
                    hostport = after_at.split("/", 1)[0]
                    if ":" in hostport:
                        final_port = int(hostport.split(":")[1])
                except Exception:
                    # ignore if not parseable
                    pass

        # Generate agent_id if not provided
        agent_id = b.agent_id or str(uuid4())
        
        with appdb() as conn:
            cur = conn.cursor()
            
            # Insert into agent table
            cur.execute("""
                INSERT INTO agent(agent_id, user_id, agent_name, agent_description, 
                                 create_by, create_date)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (agent_id, uid, b.name, b.description, uid))
            
            # Parse connection details
            parsed = parse_db_url(final_db_url)
            
            # Insert into db_configs table
            cur.execute("""
                INSERT INTO db_configs(agent_id, db_type, db_connection_string,
                                      db_host, db_name, db_user, db_pass, db_port,
                                      create_by, create_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (agent_id, final_dbtype, final_db_url,
                  parsed['host'], parsed['db'], parsed['user'],
                  parsed['password'], final_port, uid))
            
            conn.commit()

            # Add default schema-specific system prompt as filter
            default_schema = get_default_schema(final_dbtype)
            if final_dbtype == "sqlserver":
                prompt_content = f"This agent is connected to a SQL Server database. Always use the schema '{default_schema}' unless another schema is explicitly specified in the question."
            elif final_dbtype == "mysql":
                prompt_content = f"This agent is connected to a MySQL database. Use the database name as schema context."
            else:  # PostgreSQL
                prompt_content = f"This agent is connected to a PostgreSQL database. Always use the schema '{default_schema}' unless another schema is explicitly specified."
            
            cur.execute("""
                INSERT INTO filters(agent_id, filter_name, nlq, filter_description,
                                   create_by, create_date)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (agent_id, "Default Schema Prompt", prompt_content,
                  "Auto-generated schema guidance", uid))
            
            conn.commit()
            cur.close()

        return {
            "success": True,
            "saved": True,
            "agent_id": agent_id,
            "connectionStringPreview": _mask_url(final_db_url)
        }
    except Exception as e:
        print(f"Error creating agent: {e}")
        return {
            "success": False,
            "saved": False,
            "agent_id": None,
            "error": str(e)
        }

# ---------- System Prompts ----------
class SetPromptBody(BaseModel):
    agent_id: str
    content: str
    deactivate_others: bool = True

@app.post("/prompts")
def set_prompt(b: SetPromptBody):
    try:
        with appdb() as conn:
            cur = conn.cursor()
            
            if b.deactivate_others:
                cur.execute("""
                    UPDATE filters 
                    SET is_active=0, update_by=1, update_date=NOW() 
                    WHERE agent_id=%s
                """, (b.agent_id,))
            
            cur.execute("""
                INSERT INTO filters(agent_id, filter_name, nlq, filter_description,
                                   create_by, create_date)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (b.agent_id, "Custom System Prompt", b.content, 
                  "User-defined system prompt", 1))
            
            conn.commit()
            cur.close()
            
        return {"success": True}
    except Exception as e:
        print(f"Error setting prompt: {e}")
        return {"success": False, "error": str(e)}

# ---------- QnA ----------
class AddQnABody(BaseModel):
    agent_id: str
    question: str
    sql_query: str
    answer_preview: Optional[str] = None
    user_id: Optional[str] = None
    favorite_name: Optional[str] = None

@app.post("/qna")
def add_qna(b: AddQnABody):
    """Add to favourites via stored procedure sp_add_favorite_query.
    - p_agent_id: UUID string (DB side updated to accept string UUID)
    - p_favorite_name: favourite name (prompted in UI) or fallback to NLQ (150 chars)
    - p_favorite_query: generated SQL
    - p_description: NLQ (as requested)
    - p_created_by: numeric user id (fallback 1)
    - p_created_purpose: NULL
    """
    try:
        with appdb() as conn:
            cur = conn.cursor()
            fav_name = (b.favorite_name or b.question or "")[:150]
            desc = b.question or ""
            created_by = 1
            try:
                if b.user_id is not None and str(b.user_id).isdigit():
                    created_by = int(str(b.user_id))
            except Exception:
                created_by = 1

            # CALL sp_add_favorite_query(agent_id, favorite_name, favorite_query, description, created_by, created_purpose=NULL)
            cur.callproc(
                "sp_add_favorite_query",
                [b.agent_id, fav_name, b.sql_query, desc, created_by, None]
            )
            conn.commit()
            cur.close()
        return {"success": True}
    except Exception as e:
        print(f"Error adding QnA: {e}")
        return {"success": False, "error": str(e)}

# ---------- History ----------
@app.get("/history/{agent_id}")
def history(agent_id: str, limit: int = 50):
    """Get conversation history - using favorite_queries as history log"""
    try:
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT favorite_query_id, favorite_name, favorite_query, 
                       description, created_at, created_by
                FROM favorite_queries
                WHERE agent_id=%s AND is_active=1
                ORDER BY created_at DESC
                LIMIT %s
            """, (agent_id, limit))
            rows = cur.fetchall()
            cur.close()
        
        # Format as conversations
        conversations = []
        for r in rows:
            conversations.append({
                "session_id": str(r['favorite_query_id']),
                "messages": [
                    {
                        "id": str(r['favorite_query_id']) + "_q",
                        "ts": r['created_at'].isoformat() if r['created_at'] else None,
                        "role": "user",
                        "content": r['favorite_name'],
                        "sql_query": None
                    },
                    {
                        "id": str(r['favorite_query_id']) + "_a",
                        "ts": r['created_at'].isoformat() if r['created_at'] else None,
                        "role": "assistant",
                        "content": r['description'] or "Query executed",
                        "sql_query": r['favorite_query']
                    }
                ]
            })
        
        return {"conversations": conversations}
    except Exception as e:
        print(f"Error getting history: {e}")
        return {"conversations": []}

# ---------- Get Agents ----------
@app.get("/agents/{user_id}")
def get_agents(user_id: str):
    try:
        uid = int(str(user_id)) if str(user_id).isdigit() else 1
        if uid is None:
            return []
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT a.agent_id as id, a.agent_name as name, 
                       a.agent_description as description,
                       a.create_date as created_at, a.update_date as trained_at,
                       dc.db_connection_string as db_url, dc.db_type as dbtype
                FROM agent a
                LEFT JOIN db_configs dc ON a.agent_id = dc.agent_id AND dc.is_active=1
                WHERE a.user_id=%s AND a.is_active=1
                ORDER BY a.create_date DESC
            """, (uid,))
            rows = cur.fetchall()
            cur.close()
            
        return [
            {
                "id": r['id'],
                "name": r['name'],
                "description": r['description'],
                "db_url": r['db_url'],
                "created_at": r['created_at'],
                "trained_at": r['trained_at'],
                "dbtype": r['dbtype']
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Error getting agents: {e}")
        return []

# ---------- Get Agent Details ----------
@app.get("/agents/detail/{agent_id}")
def get_agent_detail(agent_id: str):
    try:
        with appdb() as conn:
            agent = get_agent(conn, agent_id)
            if not agent:
                return {"error": "Agent not found"}
            
            # Get widget count for this agent
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) as widget_count
                FROM dashboard_widgets dw
                INNER JOIN dashboards d ON dw.dashboard_id = d.dashboard_id
                WHERE d.user_id IN (SELECT user_id FROM agent WHERE agent_id=%s)
                  AND dw.is_active=1
            """, (agent_id,))
            widget_count = cur.fetchone()[0]
            cur.close()
            
            agent['widget_count'] = widget_count
            return agent
    except Exception as e:
        print(f"Error getting agent detail: {e}")
        return {"error": str(e)}

# ---------- Dashboards ----------
class CreateDashboardBody(BaseModel):
    user_id: str
    dashboard_name: str
    dashboard_description: Optional[str] = None
    is_public: bool = False

@app.post("/dashboards")
def create_dashboard(b: CreateDashboardBody):
    try:
        with appdb() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO dashboards(user_id, dashboard_name, dashboard_description, 
                                      is_public, create_by, create_date)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (b.user_id, b.dashboard_name, b.dashboard_description, 
                  b.is_public, b.user_id))
            conn.commit()
            dashboard_id = cur.lastrowid
            cur.close()
            
        return {"success": True, "dashboard_id": dashboard_id}
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return {"success": False, "error": str(e)}

@app.get("/dashboards/{user_id}")
def get_dashboards(user_id: str):
    try:
        uid = int(str(user_id)) if str(user_id).isdigit() else 1
        if uid is None:
            return {"dashboards": []}
        with appdb() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT dashboard_id, dashboard_name, dashboard_description, 
                       is_public, layout_config, create_date
                FROM dashboards
                WHERE user_id=%s AND is_active=1
                ORDER BY create_date DESC
            """, (uid,))
            rows = cur.fetchall()
            cur.close()
            
        return {"dashboards": rows}
    except Exception as e:
        print(f"Error getting dashboards: {e}")
        return {"dashboards": []}

# ---------- Widgets ----------
class CreateWidgetBody(BaseModel):
    widget_name: str
    widget_type: str
    sql_query: str
    user_id: str

@app.post("/widgets")
def create_widget(b: CreateWidgetBody):
    try:
        with appdb() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO widgets(widget_name, widget_type, sql_query, 
                                   create_by, create_date)
                VALUES (%s, %s, %s, %s, NOW())
            """, (b.widget_name, b.widget_type, b.sql_query, b.user_id))
            conn.commit()
            widget_id = cur.lastrowid
            cur.close()
            
        return {"success": True, "widget_id": widget_id}
    except Exception as e:
        print(f"Error creating widget: {e}")
        return {"success": False, "error": str(e)}

# Health endpoints
@app.get("/health")
def health_check():
    """Lightweight health check that responds quickly"""
    return {
        "status": "healthy",
        "service": "python-ai",
        "timestamp": datetime.utcnow().isoformat()
    }

# Detailed health check for comprehensive status
@app.get("/health/detailed")
def detailed_health_check():
    """Comprehensive health check with database and Vanna status"""
    try:
        # Test database connection with timeout
        db_status = "unknown"
        db_error = None
        
        try:
            with appdb() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            db_status = "connected"
        except Exception as e:
            db_status = "disconnected"
            db_error = str(e)
        
        # Test Vanna status
        vanna_status = "unknown"
        vanna_error = None
        
        try:
            vanna_instance = get_vanna()
            vanna_status = "available" if vanna_instance else "unavailable"
        except Exception as e:
            vanna_status = "error"
            vanna_error = str(e)
        
        # Determine overall status
        overall_status = "healthy"
        if db_status != "connected" or vanna_status not in ["available", "unavailable"]:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "database": {
                "status": db_status,
                "error": db_error
            },
            "vanna": {
                "status": vanna_status,
                "error": vanna_error
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Alternative: Even simpler ping endpoint
@app.get("/ping")
def ping():
    """Ultra-fast ping endpoint"""
    return {"pong": True, "timestamp": datetime.utcnow().isoformat()}

# Separate endpoint to check database only
@app.get("/health/database") 
def check_database():
    """Check database connection only"""
    try:
        with appdb() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        return {"database": "connected", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"database": "disconnected", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

# Separate endpoint to check Vanna only  
@app.get("/health/vanna")
def check_vanna():
    """Check Vanna availability only"""
    try:
        vanna_instance = get_vanna()
        status = "available" if vanna_instance else "unavailable"
        return {"vanna": status, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"vanna": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

# ---------- Enhanced SQL Server utilities ----------
@app.get("/test-sqlserver-drivers")
def test_sqlserver_drivers():
    """Test available SQL Server drivers and configurations"""
    try:
        drivers = pyodbc_driver.drivers() if pyodbc_driver else []
        available_drivers = [d for d in drivers if "SQL Server" in d]
        
        return {
            "success": True,
            "available_drivers": available_drivers,
            "recommended": "ODBC Driver 17 for SQL Server" in available_drivers,
            "all_drivers": drivers
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/validate-connection-string")
def validate_connection_string(data: ConnectionData):
    """Validate and return optimized connection strings for different purposes"""
    try:
        if data.db_type.lower() == "sqlserver":
            pyodbc_str = build_sqlserver_connection_string(
                data.host, data.port, data.db_name, data.user, data.password, use_sqlalchemy=False
            )
            sqlalchemy_str = build_sqlserver_connection_string(
                data.host, data.port, data.db_name, data.user, data.password, use_sqlalchemy=True
            )
            
            return {
                "success": True,
                "pyodbc_string": pyodbc_str,
                "sqlalchemy_string": sqlalchemy_str,
                "vanna_compatible": True
            }
        else:
            return {"success": False, "message": "Only SQL Server validation supported"}
    except Exception as e:
        return {"success": False, "error": str(e)}