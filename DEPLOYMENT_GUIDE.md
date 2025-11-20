Email Sentiment Monitor - Hostinger Deployment Guide
📋 Prerequisites

Hostinger Account with:

VPS or Cloud Hosting plan
SSH access enabled
PostgreSQL database support


Local Requirements:

Git installed
SSH client
Your Microsoft Graph API credentials



🗄️ Step 1: Set Up PostgreSQL Database
Option A: Using Hostinger's Managed PostgreSQL

Log in to Hostinger hPanel
Navigate to Databases → PostgreSQL
Click Create Database
Note down:

Database name
Username
Password
Hostname
Port (usually 5432)



Option B: Using Hostinger VPS
If you have VPS access, install PostgreSQL:
bash# SSH into your VPS
ssh root@your-vps-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
In PostgreSQL prompt:
sqlCREATE DATABASE email_monitor;
CREATE USER email_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE email_monitor TO email_user;
\q
🚀 Step 2: Deploy Application to Hostinger
Connect to Your Server
bashssh your-username@your-hostinger-server
Install Required Software
bash# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
Clone Your Repository
bash# Create application directory
mkdir -p /opt/email-monitor
cd /opt/email-monitor

# Clone your repository (replace with your repo URL)
git clone https://github.com/your-username/email-monitor.git .

# Or upload files via SFTP/SCP
Configure Environment Variables
bash# Copy environment template
cp .env.example .env

# Edit environment file
nano .env
Add your configuration:
bash# Microsoft Graph API
CLIENT_ID=your_client_id_from_azure
TENANT_ID=your_tenant_id_from_azure
CLIENT_SECRET=your_client_secret_from_azure

# Mailboxes to monitor
MAILBOX_LIST=user1@yourdomain.com,user2@yourdomain.com

# PostgreSQL Database (use your Hostinger database details)
DATABASE_URL=postgresql://email_user:your_password@localhost:5432/email_monitor

# Or if using Hostinger managed database:
# DATABASE_URL=postgresql://username:password@hostname:5432/database_name

# Application Settings
POLL_INTERVAL=30
DEBUG_CC=False
PORT=8050
ADMIN_PORT=8051
Save and exit (Ctrl+X, then Y, then Enter)
📁 Step 3: Organize Project Structure
Ensure your project structure looks like this:
/opt/email-monitor/
├── monitor/
│   ├── Dockerfile
│   ├── graph_delegate_monitor.py
│   ├── inference_local.py
│   └── requirements.txt
├── dashboard/
│   ├── Dockerfile
│   ├── app.py
│   ├── admin.py
│   └── requirements.txt
├── db_config.py
├── migrate_sqlite_to_postgres.py
├── docker-compose.yml
├── .env
└── data/
🔄 Step 4: Migrate Existing Data (Optional)
If you have existing SQLite data:
bash# Install Python dependencies locally
pip3 install psycopg2-binary python-dotenv

# Run migration script
python3 migrate_sqlite_to_postgres.py
🐳 Step 5: Build and Start Services
bash# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Check if services are running
docker-compose ps

# View logs
docker-compose logs -f
Expected output:
NAME                    STATUS              PORTS
email_postgres          running             5432/tcp
email_monitor           running
email_dashboard         running             0.0.0.0:8050->8050/tcp
email_admin_dashboard   running             0.0.0.0:8051->8051/tcp
🔒 Step 6: Configure Firewall and Reverse Proxy
Open Required Ports
bash# Allow PostgreSQL (if external access needed)
sudo ufw allow 5432/tcp

# Allow dashboard ports
sudo ufw allow 8050/tcp
sudo ufw allow 8051/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
Set Up Nginx Reverse Proxy (Recommended)
bash# Install Nginx
sudo apt install nginx -y

# Create configuration file
sudo nano /etc/nginx/sites-available/email-monitor
Add this configuration:
nginxserver {
    listen 80;
    server_name your-domain.com;

    # Main Dashboard
    location / {
        proxy_pass http://localhost:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Admin Dashboard
    location /admin {
        proxy_pass http://localhost:8051;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
Enable the site:
bash# Enable configuration
sudo ln -s /etc/nginx/sites-available/email-monitor /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
Set Up SSL with Let's Encrypt
bash# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
✅ Step 7: Verify Deployment
Check Services
bash# Check all containers are running
docker-compose ps

# Check PostgreSQL connection
docker-compose exec postgres psql -U postgres -d email_monitor -c "SELECT COUNT(*) FROM processed;"

# View monitor logs
docker-compose logs -f monitor

# View dashboard logs
docker-compose logs -f dashboard
Access Your Dashboards

Main Dashboard: http://your-domain.com or http://your-server-ip:8050
Admin Dashboard: http://your-domain.com/admin or http://your-server-ip:8051

🔧 Step 8: Set Up Auto-Start on Boot
bash# Create systemd service
sudo nano /etc/systemd/system/email-monitor.service
Add this content:
ini[Unit]
Description=Email Monitor Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/email-monitor
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
Enable the service:
bashsudo systemctl enable email-monitor
sudo systemctl start email-monitor
📊 Monitoring and Maintenance
View Logs
bash# All services
docker-compose logs -f

# Specific service
docker-compose logs -f monitor
docker-compose logs -f dashboard

# Last 100 lines
docker-compose logs --tail=100
Restart Services
bash# Restart all
docker-compose restart

# Restart specific service
docker-compose restart monitor
docker-compose restart dashboard
Update Application
bashcd /opt/email-monitor

# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
Database Backup
bash# Create backup script
nano /opt/email-monitor/backup.sh
Add:
bash#!/bin/bash
BACKUP_DIR="/opt/email-monitor/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

docker-compose exec -T postgres pg_dump -U postgres email_monitor > "$BACKUP_DIR/backup_$DATE.sql"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete

echo "Backup completed: backup_$DATE.sql"
Make it executable and add to cron:
bashchmod +x /opt/email-monitor/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add this line:
0 2 * * * /opt/email-monitor/backup.sh
🐛 Troubleshooting
Cannot Connect to PostgreSQL
bash# Check PostgreSQL is running
docker-compose ps postgres

# Check connection from monitor
docker-compose exec monitor python3 -c "from db_config import test_connection; test_connection()"

# Check PostgreSQL logs
docker-compose logs postgres
Dashboard Not Accessible
bash# Check if port is open
sudo netstat -tulpn | grep 8050

# Check dashboard logs
docker-compose logs dashboard

# Restart dashboard
docker-compose restart dashboard
High Memory Usage
bash# Check resource usage
docker stats

# Limit container resources (edit docker-compose.yml)
# Add under each service:
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
📞 Support
For issues:

Check logs: docker-compose logs -f
Verify environment variables: cat .env
Test database connection: python3 db_config.py
Check Hostinger documentation for VPS/database specifics