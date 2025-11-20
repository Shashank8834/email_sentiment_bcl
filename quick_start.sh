#!/bin/bash
# quick_start.sh - Automated setup script for Email Monitor on Hostinger

set -e  # Exit on error

echo "================================================"
echo "Email Sentiment Monitor - Quick Start Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_warning "This script should not be run as root. Run as regular user with sudo access."
   exit 1
fi

# Check for required commands
check_dependencies() {
    print_info "Checking dependencies..."
    
    commands=("docker" "docker-compose" "git")
    missing=()
    
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_info "Please install missing dependencies first."
        exit 1
    fi
    
    print_success "All dependencies found"
}

# Create .env file if it doesn't exist
setup_env_file() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from template"
            print_warning "⚠️  IMPORTANT: Edit .env file with your actual credentials before continuing!"
            echo ""
            echo "Required variables:"
            echo "  - CLIENT_ID"
            echo "  - TENANT_ID"
            echo "  - CLIENT_SECRET"
            echo "  - DATABASE_URL"
            echo "  - MAILBOX_LIST"
            echo ""
            read -p "Press Enter when you've updated .env file..."
        else
            print_error ".env.example not found. Cannot create .env file."
            exit 1
        fi
    else
        print_success ".env file found"
    fi
}

# Validate environment variables
validate_env() {
    print_info "Validating environment variables..."
    
    source .env
    
    required_vars=("CLIENT_ID" "TENANT_ID" "CLIENT_SECRET" "DATABASE_URL")
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=($var)
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        print_error "Missing required environment variables: ${missing_vars[*]}"
        print_info "Please update your .env file"
        exit 1
    fi
    
    print_success "Environment variables validated"
}

# Create required directories
setup_directories() {
    print_info "Creating required directories..."
    
    mkdir -p data
    mkdir -p data/model
    mkdir -p backups
    
    print_success "Directories created"
}

# Test database connection
test_database() {
    print_info "Testing PostgreSQL connection..."
    
    if python3 -c "from db_config import test_connection; exit(0 if test_connection() else 1)" 2>/dev/null; then
        print_success "Database connection successful"
    else
        print_warning "Could not test database connection (db_config.py may not be available yet)"
        print_info "This is okay - Docker containers will test the connection when they start"
    fi
}

# Initialize database schema
init_database() {
    print_info "Initializing database schema..."
    
    if python3 -c "from db_config import ensure_schema, init_connection_pool; init_connection_pool(); ensure_schema()" 2>/dev/null; then
        print_success "Database schema initialized"
    else
        print_warning "Could not initialize schema locally"
        print_info "Schema will be initialized when monitor container starts"
    fi
}

# Migrate existing SQLite data (optional)
migrate_data() {
    if [ -f "data/monitor.db" ]; then
        echo ""
        read -p "SQLite database found. Migrate data to PostgreSQL? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Running migration..."
            python3 migrate_sqlite_to_postgres.py
            print_success "Migration completed"
        else
            print_info "Skipping migration"
        fi
    fi
}

# Build Docker images
build_containers() {
    print_info "Building Docker containers..."
    
    if docker-compose build; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Start services
start_services() {
    print_info "Starting services..."
    
    if docker-compose up -d; then
        print_success "Services started"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Check service health
check_services() {
    print_info "Checking service health..."
    sleep 5
    
    services=("postgres" "monitor" "dashboard" "admin")
    
    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            print_success "$service is running"
        else
            print_error "$service is not running"
            docker-compose logs $service
        fi
    done
}

# Display access information
show_access_info() {
    echo ""
    echo "================================================"
    print_success "Setup Complete!"
    echo "================================================"
    echo ""
    echo "🎉 Your Email Sentiment Monitor is now running!"
    echo ""
    echo "Access your dashboards:"
    echo "  📊 Main Dashboard:  http://localhost:8050"
    echo "  ⚙️  Admin Dashboard: http://localhost:8051"
    echo ""
    echo "Useful commands:"
    echo "  📋 View logs:        docker-compose logs -f"
    echo "  🔄 Restart:          docker-compose restart"
    echo "  ⛔ Stop:             docker-compose down"
    echo "  📊 Check status:     docker-compose ps"
    echo ""
    echo "Database:"
    echo "  📦 PostgreSQL is running on port 5432"
    echo "  💾 Data is persisted in Docker volume"
    echo ""
}

# Main execution
main() {
    echo "Starting setup process..."
    echo ""
    
    check_dependencies
    setup_env_file
    validate_env
    setup_directories
    test_database
    init_database
    migrate_data
    build_containers
    start_services
    check_services
    show_access_info
}

# Run main function
main

print_success "Setup script completed successfully!"