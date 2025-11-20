# migrate_sqlite_to_postgres.py
# Script to migrate data from SQLite to PostgreSQL

import sqlite3
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Import PostgreSQL configuration
from db_config import get_db_connection, ensure_schema, init_connection_pool, test_connection

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/data/monitor.db")

def migrate_data():
    """Migrate data from SQLite to PostgreSQL"""
    
    print("="*60)
    print("🔄 SQLite to PostgreSQL Migration Tool")
    print("="*60)
    
    # Test PostgreSQL connection
    print("\n📡 Testing PostgreSQL connection...")
    if not test_connection():
        print("❌ Cannot connect to PostgreSQL. Please check your DATABASE_URL.")
        sys.exit(1)
    
    # Initialize PostgreSQL
    print("\n🏗️  Initializing PostgreSQL schema...")
    init_connection_pool()
    ensure_schema()
    
    # Check if SQLite database exists
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"\n⚠️  SQLite database not found at: {SQLITE_DB_PATH}")
        print("This is okay if you're starting fresh with PostgreSQL.")
        sys.exit(0)
    
    # Connect to SQLite
    print(f"\n📂 Opening SQLite database: {SQLITE_DB_PATH}")
    try:
        sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        sqlite_cur = sqlite_conn.cursor()
    except Exception as e:
        print(f"❌ Failed to connect to SQLite: {e}")
        sys.exit(1)
    
    # Count records in SQLite
    try:
        sqlite_cur.execute("SELECT COUNT(*) FROM processed")
        total_records = sqlite_cur.fetchone()[0]
        print(f"📊 Found {total_records:,} records in SQLite")
    except Exception as e:
        print(f"❌ Failed to count SQLite records: {e}")
        sqlite_conn.close()
        sys.exit(1)
    
    if total_records == 0:
        print("ℹ️  No records to migrate.")
        sqlite_conn.close()
        sys.exit(0)
    
    # Fetch all records from SQLite
    print("\n📥 Reading records from SQLite...")
    try:
        sqlite_cur.execute("""
            SELECT message_id, mailbox, sender, receivers, cc, subject,
                   received_dt, web_link, final_label, prob_neg, 
                   sender_domain, processed_at
            FROM processed
        """)
        records = sqlite_cur.fetchall()
        print(f"✅ Successfully read {len(records):,} records")
    except Exception as e:
        print(f"❌ Failed to read from SQLite: {e}")
        sqlite_conn.close()
        sys.exit(1)
    
    sqlite_conn.close()
    
    # Migrate to PostgreSQL
    print(f"\n📤 Migrating {len(records):,} records to PostgreSQL...")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    with get_db_connection() as pg_conn:
        with pg_conn.cursor() as pg_cur:
            for i, record in enumerate(records, 1):
                try:
                    (message_id, mailbox, sender, receivers, cc, subject,
                     received_dt, web_link, final_label, prob_neg, 
                     sender_domain, processed_at) = record
                    
                    pg_cur.execute("""
                        INSERT INTO processed
                          (message_id, mailbox, sender, receivers, cc, subject,
                           received_dt, web_link, final_label, prob_neg,
                           sender_domain, processed_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (message_id) DO NOTHING
                    """, (
                        message_id, mailbox, sender, receivers, cc, subject,
                        received_dt, web_link, final_label, prob_neg,
                        sender_domain, processed_at
                    ))
                    
                    if pg_cur.rowcount > 0:
                        migrated += 1
                    else:
                        skipped += 1
                    
                    # Progress indicator
                    if i % 100 == 0:
                        print(f"   Progress: {i:,}/{len(records):,} ({i/len(records)*100:.1f}%)")
                
                except Exception as e:
                    errors += 1
                    print(f"⚠️  Error migrating record {i}: {e}")
                    if errors > 10:
                        print("❌ Too many errors, stopping migration.")
                        break
    
    # Summary
    print("\n" + "="*60)
    print("✅ Migration Complete!")
    print("="*60)
    print(f"📊 Total records processed: {len(records):,}")
    print(f"✅ Successfully migrated: {migrated:,}")
    print(f"⏭️  Skipped (duplicates): {skipped:,}")
    print(f"❌ Errors: {errors:,}")
    print("="*60)
    
    # Verify migration
    print("\n🔍 Verifying PostgreSQL data...")
    with get_db_connection() as pg_conn:
        with pg_conn.cursor() as pg_cur:
            pg_cur.execute("SELECT COUNT(*) FROM processed")
            pg_count = pg_cur.fetchone()[0]
            print(f"📊 PostgreSQL now has {pg_count:,} records")
    
    if pg_count == total_records:
        print("✅ All records successfully migrated!")
    elif pg_count > 0:
        print(f"⚠️  PostgreSQL has {pg_count:,} records (expected {total_records:,})")
        print("   This might be due to duplicates being skipped.")
    else:
        print("❌ Migration may have failed. Please check the errors above.")

if __name__ == "__main__":
    try:
        migrate_data()
    except KeyboardInterrupt:
        print("\n\n⛔ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)