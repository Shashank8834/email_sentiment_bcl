# db_config.py
# PostgreSQL database configuration and connection management

import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from urllib.parse import urlparse

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Parse database URL if provided (common in cloud hosting)
if DATABASE_URL:
    url = urlparse(DATABASE_URL)
    DB_CONFIG = {
        "host": url.hostname,
        "port": url.port or 5432,
        "database": url.path[1:],  # Remove leading slash
        "user": url.username,
        "password": url.password,
    }
else:
    # Fallback to individual environment variables
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "email_monitor"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }

# Connection pool for better performance
connection_pool = None

def init_connection_pool(min_conn=1, max_conn=10):
    """Initialize PostgreSQL connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            min_conn,
            max_conn,
            **DB_CONFIG
        )
        if connection_pool:
            print("✅ Connection pool created successfully")
    except Exception as e:
        print(f"❌ Error creating connection pool: {e}")
        raise

def get_connection():
    """Get a connection from the pool"""
    if connection_pool:
        return connection_pool.getconn()
    else:
        return psycopg2.connect(**DB_CONFIG)

def release_connection(conn):
    """Return a connection to the pool"""
    if connection_pool:
        connection_pool.putconn(conn)
    else:
        conn.close()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)

def ensure_schema():
    """Create the processed table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS processed (
        message_id TEXT PRIMARY KEY,
        mailbox TEXT,
        sender TEXT,
        receivers TEXT,
        cc TEXT,
        subject TEXT,
        received_dt TIMESTAMP,
        web_link TEXT,
        final_label TEXT,
        prob_neg REAL,
        sender_domain TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes for better query performance
    CREATE INDEX IF NOT EXISTS idx_processed_mailbox ON processed(mailbox);
    CREATE INDEX IF NOT EXISTS idx_processed_sender_domain ON processed(sender_domain);
    CREATE INDEX IF NOT EXISTS idx_processed_final_label ON processed(final_label);
    CREATE INDEX IF NOT EXISTS idx_processed_received_dt ON processed(received_dt);
    CREATE INDEX IF NOT EXISTS idx_processed_at ON processed(processed_at);
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                print("✅ Database schema initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing schema: {e}")
        raise

def test_connection():
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                print(f"✅ PostgreSQL connection successful: {version[0]}")
                return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    print(f"Database: {DB_CONFIG['database']}")
    print(f"Host: {DB_CONFIG['host']}")
    print(f"Port: {DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    
    if test_connection():
        print("\nInitializing schema...")
        ensure_schema()
        print("\n✅ Database setup complete!")
    else:
        print("\n❌ Database setup failed!")