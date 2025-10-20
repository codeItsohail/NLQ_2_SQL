# === add imports near top ===
import os, time, uuid, re
from fastapi import HTTPException
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from datetime import datetime
from fastapi import BackgroundTasks
import pandas as pd
import pyodbc
import mysql.connector
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

# --- Vanna with OpenAI + ChromaDB (local vector store) ---
class MyVanna122(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        if config is None:
            config = {}
        config['path'] = config.get('path', './chroma_db')
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

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
        return super().generate_sql(question)

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
                return {"success": False, "error": "MySQL connector not available. Install mysql-connector-python"}
            
            conn = mysql.connector.connect(
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
                return {"success": False, "error": "pyodbc not available. Install pyodbc"}
                
            # Test with pyodbc first
            conn_str = build_sqlserver_connection_string(
                data.host, data.port, data.db_name, data.user, data.password, use_sqlalchemy=False
            )
            conn = pyodbc.connect(conn_str, timeout=10)
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
    if not APP_DB_URL:
        raise RuntimeError("APP_DATABASE_URL not set")
    return psycopg2.connect(_normalize_pg_url(APP_DB_URL), sslmode="require")

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
        return mysql.connector.connect(host=p["host"], port=p["port"], database=p["db"],
                                       user=p["user"], password=p["password"], connection_timeout=10)
    elif p["scheme"] == "sqlserver":
        if pyodbc_driver is None:
            raise RuntimeError("pyodbc not available")
        conn_str = build_sqlserver_connection_string(
            p["host"], p["port"], p["db"], p["user"], p["password"], use_sqlalchemy=False
        )
        return pyodbc.connect(conn_str, timeout=10)
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
    with conn.cursor() as cur:
        cur.execute("SELECT id,user_id,name,description,db_url,trained_at FROM ai_agents WHERE id=%s", (agent_id,))
        r = cur.fetchone()
    if not r: return None
    return {"id": r[0], "user_id": r[1], "name": r[2], "description": r[3], "db_url": r[4], "trained_at": r[5]}

def info_schema_text(db_url: str) -> str:
    """Enhanced schema extraction with SQL Server support"""
    p = parse_db_url(db_url)
    with connect_target(db_url) as conn:
        cur = conn.cursor()
        if p["scheme"] == "postgresql":
            cur.execute("""
              SELECT table_schema, table_name, column_name, data_type
              FROM information_schema.columns
              WHERE table_schema NOT IN ('pg_catalog','information_schema')
              ORDER BY table_schema, table_name, ordinal_position
            """)
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
              ORDER BY t.name, c.column_id
            """)
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

def log_exec(conn, agent_id, user_id, question, sql, success, err, row_count, dur_ms):
    with conn.cursor() as cur:
        cur.execute("""
          INSERT INTO execution_logs(agent_id,user_id,question,generated_sql,success,error,row_count,duration_ms)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (agent_id, user_id, question, sql, success, err, row_count, dur_ms))
    conn.commit()

def ensure_session(conn, agent_id, user_id, session_id) -> str:
    if session_id:
        return session_id
    sid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO agent_sessions(id, agent_id, user_id) VALUES (%s,%s,%s)",
                    (sid, agent_id, user_id))
    conn.commit()
    return sid

def store_message(conn, session_id: str, role: str, content: str, sql_query: str = None):
    """Store message in agent_messages table"""
    with conn.cursor() as cur:
        meta = {"sql_query": sql_query} if sql_query else {}
        cur.execute("""
            INSERT INTO agent_messages (session_id, role, content, meta)
            VALUES (%s, %s, %s, %s)
        """, (session_id, role, content, json.dumps(meta)))
    conn.commit()

def log_query(conn, user_id: str, question: str, sql_generated: str, error: str = None, success: bool = True, agent_id: str = None):
    """Log query to query_logs table"""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO query_logs (user_id, question, sql_generated, error, success, agent_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, question, sql_generated, error, success, agent_id))
    conn.commit()

def _mask_url(url: str) -> str:
    # mask the password portion between ":" and "@" once
    if ":" in url and "@" in url:
        before, after = url.split("@", 1)
        if ":" in before:
            head, _pwd = before.rsplit(":", 1)
            return f"{head}:****@{after}"
    return url

def apply_system_prompts(v, conn, agent_id: str):
    """Load active system prompts from DB into Vanna instance."""
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM system_prompts WHERE agent_id=%s AND is_active=TRUE", (agent_id,))
        rows = cur.fetchall()
    for (content,) in rows:
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
        }
    else:
        cfg = {
            "api_key": api_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
        with appdb() as conn, conn.cursor() as cur:
            # Check if agent exists
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"error": "Agent not found", "run_id": None, "status": "failed"}
                
            cur.execute(
                "INSERT INTO training_runs (id,agent_id,status,progress,message) VALUES (%s,%s,'queued',0,'Queued')",
                (run_id, body.agent_id),
            )
            conn.commit()
        background_tasks.add_task(_train_job, run_id, body.agent_id)
        return {"run_id": run_id, "status": "queued", "progress": 0}
    except Exception as e:
        print(f"Error starting training: {e}")
        return {"error": str(e), "run_id": None, "status": "failed"}

def _train_job(run_id: str, agent_id: str):
    def upd(conn, prog, status, msg):
        with conn.cursor() as c:
            c.execute("UPDATE training_runs SET progress=%s,status=%s,message=%s WHERE id=%s",
                      (prog, status, msg, run_id))
            c.execute("INSERT INTO training_logs(run_id,progress,message) VALUES (%s,%s,%s)",
                      (run_id, prog, msg))
        conn.commit()

    try:
        with appdb() as conn:
            upd(conn, 5, "running", "Loading agent")
            agent = get_agent(conn, agent_id)
            if not agent:
                upd(conn, 100, "failed", "Agent not found")
                return
            db_url = agent["db_url"]
            
            upd(conn, 15, "running", "Reading database schema")
            try:
                schema_text = info_schema_text(db_url)
                if not schema_text.strip():
                    upd(conn, 100, "failed", "No schema found in database")
                    return
            except Exception as e:
                upd(conn, 100, "failed", f"Failed to read schema: {str(e)}")
                return

            upd(conn, 35, "running", "Initializing AI training (Vanna/OpenAI + ChromaDB)")
            v = get_vanna()
            if not v:
                upd(conn, 100, "failed", "Vanna not available - check OpenAI API key")
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
                upd(conn, 100, "failed", f"Failed to connect Vanna to database: {e}")
                return
            
            upd(conn, 55, "running", "Training on database schema")
            try:
                v.train(ddl=schema_text)
            except Exception as e:
                upd(conn, 100, "failed", f"Failed to train on schema: {e}")
                return

            upd(conn, 75, "running", "Loading validated Q&A pairs")
            with conn.cursor() as cq:
                cq.execute("SELECT question, sql_query FROM qna_chunks WHERE agent_id=%s ORDER BY qna_chunks_id", (agent_id,))
                pairs = cq.fetchall()
                
            total = max(1, len(pairs))
            for i,(q,s) in enumerate(pairs, start=1):
                try:
                    v.train(question=q, sql=s)
                    upd(conn, 75 + int(20*i/total), "running", f"Training Q&A {i}/{len(pairs)}")
                except Exception as e:
                    print(f"Error training Q&A pair {i}: {e}")

            with conn.cursor() as cu:
                cu.execute("UPDATE ai_agents SET trained_at=NOW() WHERE id=%s", (agent_id,))
                conn.commit()
            upd(conn, 100, "succeeded", "Training complete. You can start asking questions.")
            
    except Exception as e:
        print(f"Training job error: {e}")
        with appdb() as conn:
            upd(conn, 100, "failed", f"Training failed: {e}")

@app.get("/train/status/{run_id}")
def train_status(run_id: str):
    try:
        with appdb() as conn, conn.cursor() as cur:
            cur.execute("SELECT agent_id,status,progress,message,started_at,finished_at FROM training_runs WHERE id=%s", (run_id,))
            r = cur.fetchone()
            if not r: return {"error": "not_found"}
            return {"run_id": run_id, "agent_id": r[0], "status": r[1], "progress": r[2],
                    "message": r[3], "started_at": r[4], "finished_at": r[5]}
    except Exception as e:
        return {"error": str(e)}

# ---------- ASK (generate SQL -> safe execute) ----------
@app.post("/ask")
def ask(body: AskBody):
    t0 = time.time()
    try:
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"answer":"Agent not found","sql":"","data":{"columns":[],"rows":[]}}
            db_url = agent["db_url"]
            
            session_id = ensure_session(conn, body.agent_id, body.user_id, body.session_id)
            store_message(conn, session_id, "user", body.question)

        # Set up Vanna with proper database connection
        v = get_vanna()
        if not v:
            error_msg = "Vanna is not available (OpenAI/Chroma import or config error)"
            with appdb() as conn:
                log_exec(conn, body.agent_id, body.user_id, body.question, "", False, error_msg, 0, int((time.time()-t0)*1000))
                log_query(conn, body.user_id or "anonymous", body.question, "", error_msg, False, body.agent_id)
            return {"answer": f"Failed: {error_msg}", "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        set_vanna_collection(v, body.agent_id)

        with appdb() as conn2:
            cur = conn2.cursor()
            cur.execute("SELECT content FROM system_prompts WHERE agent_id=%s AND is_active=TRUE", (body.agent_id,))
            for (content,) in cur.fetchall():
                v.set_system_prompt(content)

        # Connect to target database
        try:
            vanna_conn_str = get_vanna_connection_string(db_url)
            v.connect_to_database(vanna_conn_str)
        except Exception as e:
            error_msg = f"Failed to connect to database: {e}"
            with appdb() as conn:
                log_exec(conn, body.agent_id, body.user_id, body.question, "", False, error_msg, 0, int((time.time()-t0)*1000))
                log_query(conn, body.user_id or "anonymous", body.question, "", error_msg, False, body.agent_id)
            return {"answer": error_msg, "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        # Generate SQL
        sql = ""
        try:
            sql = v.generate_sql(body.question)
        except Exception as e:
            error_msg = f"SQL generation failed: {str(e)}"
            with appdb() as conn:
                log_exec(conn, body.agent_id, body.user_id, body.question, "", False, error_msg, 0, int((time.time()-t0)*1000))
                log_query(conn, body.user_id or "anonymous", body.question, "", error_msg, False, body.agent_id)
            return {"answer": error_msg, "sql":"", "data":{"columns":[],"rows":[]}, "session_id": session_id}

        columns, rows, err = [], [], None
        if body.execute and sql.strip():
            try:
                columns, rows = run_sql_preview(db_url, sql, body.limit)
            except Exception as ex:
                err = str(ex)

        dur_ms = int((time.time()-t0)*1000)
        answer = "Query executed successfully." if not err else f"Execution error: {err}"
        
        with appdb() as conn:
            log_exec(conn, body.agent_id, body.user_id, body.question, sql, err is None, err, len(rows or []), dur_ms)
            log_query(conn, body.user_id or "anonymous", body.question, sql, err, err is None, body.agent_id)
            store_message(conn, session_id, "assistant", answer, sql)

        return {"answer": answer, "sql": sql, "data": {"columns": columns, "rows": rows}, "session_id": session_id}
        
    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        return {"answer": f"Internal error: {str(e)}", "sql": "", "data": {"columns": [], "rows": []}}

# ---------- EXECUTE edited SQL (no LLM) ----------
@app.post("/execute")
def execute_sql(body: ExecuteBody):
    t0 = time.time()
    try:
        # Locate agent and target DB URL
        with appdb() as conn:
            agent = get_agent(conn, body.agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found", "columns": [], "rows": [], "sql": body.sql}
            db_url = agent["db_url"]

        # Execute SQL preview safely with limit/top enforcement
        try:
            columns, rows = run_sql_preview(db_url, body.sql, body.limit)
            success = True
            err = None
        except Exception as ex:
            columns, rows = [], []
            success = False
            err = str(ex)

        dur_ms = int((time.time() - t0) * 1000)

        # Best-effort logging (non-fatal)
        try:
            with appdb() as conn:
                log_exec(conn, body.agent_id, body.user_id, "[Manual SQL]", body.sql, success, err, len(rows or []), dur_ms)
        except Exception:
            pass

        return {
            "success": success,
            "error": err,
            "columns": columns,
            "rows": rows,
            "sql": body.sql
        }

    except Exception as e:
        return {"success": False, "error": str(e), "columns": [], "rows": [], "sql": body.sql}
 
# ---------- FEEDBACK ----------
@app.post("/feedback")
def feedback(body: FeedbackBody):
    try:
        with appdb() as conn, conn.cursor() as cur:
            cur.execute("""
              INSERT INTO training_data (question, sql_query, answer, feedback, agent_id, user_id)
              VALUES (%s,%s,%s,%s,%s,%s)
            """, (body.question, body.sql, body.answer, body.valid, body.agent_id, body.user_id))
            if body.valid:
                cur.execute("""
                  INSERT INTO qna_chunks (question, sql_query, answer_preview, embedding, agent_id, user_id)
                  VALUES (%s,%s,%s,%s,%s,%s)
                """, (body.question, body.sql, body.answer or "", None, body.agent_id, body.user_id))
            conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error in feedback endpoint: {e}")
        return {"success": False, "error": str(e)}

# ----------Create Agents ----------
@app.post("/agents")
def create_agent(b: CreateAgentBody):
    try:
        # basic validation
        if not b.user_id or not b.name:
            raise HTTPException(status_code=400, detail="user_id and name are required")
        if len(b.name) > 255:
            raise HTTPException(status_code=400, detail="Agent name must be less than 255 characters")

        # Build db_url if not provided
        final_db_url = b.db_url
        final_port = b.port
        final_dbtype = (b.dbType or "").lower() if b.dbType else None

        if not final_db_url:
            if not (final_dbtype and b.host and b.dbName and b.user is not None and b.password is not None):
                raise HTTPException(
                    status_code=400,
                    detail="When db_url is not provided, dbType, host, dbName, user, password (and optionally port) are required"
                )
            final_db_url = build_db_url(final_dbtype, b.host, b.port or 5432, b.dbName, b.user, b.password)

            # if port was omitted, infer what we used
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
        
        # Persist
        with appdb() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_agents(id, user_id, name, description, dbtype, port, db_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (agent_id, b.user_id, b.name, b.description, final_dbtype, final_port, final_db_url),
            )
            conn.commit()

            # 🚀 Add default SQL Server system prompt
            if final_dbtype == "sqlserver":
                cur.execute(
                    """
                    INSERT INTO system_prompts(agent_id, content, is_active)
                    VALUES (%s, %s, TRUE)
                    """,
                    (
                        agent_id,
                        "This agent is connected to a SQL Server database. "
                        "Use the schema 'dbo' unless another schema is explicitly specified."
                    )
                )
                conn.commit()

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
        with appdb() as conn, conn.cursor() as cur:
            if b.deactivate_others:
                cur.execute("UPDATE system_prompts SET is_active=FALSE WHERE agent_id=%s", (b.agent_id,))
            cur.execute("INSERT INTO system_prompts(agent_id, content, is_active) VALUES (%s,%s,TRUE)",
                        (b.agent_id, b.content))
            conn.commit()
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

@app.post("/qna")
def add_qna(b: AddQnABody):
    try:
        with appdb() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO qna_chunks(agent_id,user_id,question,sql_query,answer_preview,embedding)
                           VALUES (%s,%s,%s,%s,%s,NULL)""",
                           (b.agent_id, b.user_id, b.question, b.sql_query, b.answer_preview or ""))
            conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Error adding QnA: {e}")
        return {"success": False, "error": str(e)}

# ---------- History ----------
@app.get("/history/{agent_id}")
def history(agent_id: str, limit: int = 50):
    try:
        with appdb() as conn, conn.cursor() as cur:
            cur.execute("""SELECT m.created_at, s.id, m.role, m.content, m.meta, m.id
                           FROM agent_messages m
                           JOIN agent_sessions s ON s.id=m.session_id
                           WHERE s.agent_id=%s
                           ORDER BY m.created_at ASC LIMIT %s""",
                           (agent_id, limit))
            rows = cur.fetchall()
        
        # Group messages by session
        conversations = []
        current_conversation = None
        
        for r in rows:
            meta = json.loads(r[4]) if r[4] else {}
            sql_query = meta.get("sql_query")
            
            if r[2] == "user":  # New conversation starts with user message
                if current_conversation and len(current_conversation["messages"]) > 1:
                    conversations.append(current_conversation)
                current_conversation = {
                    "session_id": r[1],
                    "messages": [{
                        "id": r[5],
                        "ts": r[0].isoformat(), 
                        "role": r[2], 
                        "content": r[3],
                        "sql_query": sql_query
                    }]
                }
            elif current_conversation and r[2] == "assistant":
                current_conversation["messages"].append({
                    "id": r[5],
                    "ts": r[0].isoformat(), 
                    "role": r[2], 
                    "content": r[3],
                    "sql_query": sql_query
                })
        
        # Add the last conversation if it exists
        if current_conversation and len(current_conversation["messages"]) > 1:
            conversations.append(current_conversation)
        
        return {"conversations": conversations}
    except Exception as e:
        print(f"Error getting history: {e}")
        return {"conversations": []}

# ---------- Get Agents ----------
@app.get("/agents/{user_id}")
def get_agents(user_id: str):
    try:
        with appdb() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id, name, description, db_url, created_at, trained_at 
                           FROM ai_agents WHERE user_id=%s ORDER BY created_at DESC""", (user_id,))
            rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "db_url": r[3], 
                 "created_at": r[4], "trained_at": r[5]} for r in rows]
    except Exception as e:
        print(f"Error getting agents: {e}")
        return []

# Fast health check for Node.js service monitoring
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
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
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
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
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
        drivers = pyodbc.drivers()
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