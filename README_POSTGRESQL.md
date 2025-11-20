PostgreSQL Migration Guide
🎯 Overview
This guide covers the migration of your Email Sentiment Monitor from SQLite to PostgreSQL for production deployment on Hostinger.

📊 What Changed
Database Layer
Old: SQLite file-based database (monitor.db)
New: PostgreSQL server with connection pooling
Key Benefits
✅ Scalability: Handle more concurrent connections
✅ Performance: Better for high-volume operations
✅ Reliability: ACID compliance and better crash recovery
✅ Cloud-Ready: Works seamlessly with managed database services
✅ Concurrent Access: Multiple services can access simultaneously

📁 New Files
db_config.py - PostgreSQL connection management
Connection pooling
Context managers
Schema initialization
Database testing utilities
migrate_sqlite_to_postgres.py - Migration script
Migrates existing SQLite data
Handles duplicates
Progress tracking
Error handling
requirements.txt - Updated dependencies
Added psycopg2-binary for PostgreSQL
Added sqlalchemy for ORM support
.env.example - Environment template
PostgreSQL connection settings
Both DATABASE_URL and individual variables
🔧 Modified Files
graph_delegate_monitor.py
Replaced sqlite3 with db_config module
Changed SQL syntax from SQLite to PostgreSQL
Updated placeholder syntax: ? → %s
Added ON CONFLICT for upserts
app.py (Dashboard)
Replaced sqlite3.connect() with get_db_connection()
Updated queries for PostgreSQL compatibility
Added connection pooling initialization
admin.py (Admin Dashboard)
Same database layer updates as app.py
PostgreSQL-compatible queries
docker-compose.yml
Added PostgreSQL service
Updated environment variables
Added health checks
Configured service dependencies
🚀 Quick Start
Option 1: Fresh Installation
bash
# 1. Clone repository
git clone <your-repo>
cd email-monitor

# 2. Configure environment
cp .env.example .env
nano .env  # Add your credentials

# 3. Run quick start script
chmod +x quick_start.sh
./quick_start.sh
Option 2: Manual Setup
bash
# 1. Set up PostgreSQL (if not using managed service)
sudo apt install postgresql postgresql-contrib

# 2. Create database
sudo -u postgres createdb email_monitor
sudo -u postgres createuser email_user
sudo -u postgres psql
# In psql: GRANT ALL PRIVILEGES ON DATABASE email_monitor TO email_user;

# 3. Configure .env
DATABASE_URL=postgresql://email_user:password@localhost:5432/email_monitor

# 4. Initialize schema
python3 db_config.py

# 5. Migrate existing data (optional)
python3 migrate_sqlite_to_postgres.py

# 6. Start services
docker-compose up -d
🔄 Migration from SQLite
If you have existing SQLite data:

bash
# Set SQLite path in .env (optional)
SQLITE_DB_PATH=/data/monitor.db

# Run migration
python3 migrate_sqlite_to_postgres.py
The script will:

✅ Test PostgreSQL connection
✅ Create schema if needed
✅ Read all records from SQLite
✅ Insert into PostgreSQL (skip duplicates)
✅ Verify migration success
🔌 Connection String Formats
Format 1: DATABASE_URL (Recommended)
bash
DATABASE_URL=postgresql://username:password@hostname:port/database
Example with Hostinger:

bash
DATABASE_URL=postgresql://u123456_email:mypassword@srv123.hostinger.com:5432/u123456_email_db
Format 2: Individual Variables
bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=email_monitor
DB_USER=postgres
DB_PASSWORD=yourpassword
📊 Database Schema
sql
CREATE TABLE processed (
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

-- Indexes for performance
CREATE INDEX idx_processed_mailbox ON processed(mailbox);
CREATE INDEX idx_processed_sender_domain ON processed(sender_domain);
CREATE INDEX idx_processed_final_label ON processed(final_label);
CREATE INDEX idx_processed_received_dt ON processed(received_dt);
CREATE INDEX idx_processed_at ON processed(processed_at);
🧪 Testing
Test Database Connection
bash
python3 -c "from db_config import test_connection; test_connection()"
Test Schema Creation
bash
python3 -c "from db_config import ensure_schema, init_connection_pool; init_connection_pool(); ensure_schema()"
Check Record Count
bash
python3 -c "
from db_config import get_db_connection
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM processed')
        print(f'Total records: {cur.fetchone()[0]:,}')
"
🔍 Troubleshooting
Connection Refused
bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check if port is open
sudo netstat -tulpn | grep 5432

# Test connection manually
psql -h hostname -U username -d database_name
Authentication Failed
bash
# Check pg_hba.conf
sudo nano /etc/postgresql/15/main/pg_hba.conf

# Add line (adjust for your needs):
host    all    all    0.0.0.0/0    md5

# Restart PostgreSQL
sudo systemctl restart postgresql
Schema Not Created
bash
# Manually create schema
python3 db_config.py

# Or using Docker
docker-compose exec monitor python3 db_config.py
Migration Errors
bash
# Check PostgreSQL logs
docker-compose logs postgres

# Check migration script output
python3 migrate_sqlite_to_postgres.py 2>&1 | tee migration.log

# Retry with verbose logging
DATABASE_URL=postgresql://... python3 migrate_sqlite_to_postgres.py
📈 Performance Tips
1. Connection Pooling
Already configured in db_config.py:

python
init_connection_pool(min_conn=2, max_conn=10)
2. Batch Inserts
For large migrations, use batch processing:

python
# Instead of individual inserts
# Process in batches of 1000
3. Index Optimization
Indexes are automatically created. Monitor query performance:

sql
-- Check slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
4. Vacuum and Analyze
Regular maintenance:

bash
# Schedule in cron
docker-compose exec postgres psql -U postgres -d email_monitor -c "VACUUM ANALYZE;"
🔒 Security Best Practices
Strong Passwords: Use complex passwords for database users
Limited Access: Only allow connections from application servers
SSL/TLS: Enable for production (especially with managed databases)
Regular Backups: Automate daily backups (see deployment guide)
Monitor Access: Review PostgreSQL logs regularly
📚 Additional Resources
PostgreSQL Documentation
psycopg2 Documentation
Hostinger Database Guide
Docker PostgreSQL Image
✅ Checklist
Before going to production:

 PostgreSQL database created
 Environment variables configured
 Schema initialized (python3 db_config.py)
 Existing data migrated (if applicable)
 Docker containers built and running
 Dashboards accessible
 Monitor processing emails
 Backups configured
 SSL/TLS enabled
 Firewall rules set
🆘 Support
If you encounter issues:

Check logs: docker-compose logs -f
Test connection: python3 db_config.py
Verify environment: cat .env
Review PostgreSQL logs: docker-compose logs postgres
Check service status: docker-compose ps
Migration Complete? Head over to DEPLOYMENT_GUIDE.md for full Hostinger deployment instructions!

