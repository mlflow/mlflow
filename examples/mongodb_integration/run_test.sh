#!/bin/bash

# Genesis-Flow MongoDB Integration Test Runner
# This script provides multiple ways to test Genesis-Flow with MongoDB

set -e

echo "🎯 Genesis-Flow MongoDB Integration Test Runner"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if MongoDB is running
check_mongodb() {
    if command -v mongosh >/dev/null 2>&1; then
        if mongosh --eval "db.adminCommand('ismaster')" --quiet >/dev/null 2>&1; then
            return 0
        fi
    elif command -v mongo >/dev/null 2>&1; then
        if mongo --eval "db.adminCommand('ismaster')" --quiet >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install Genesis-Flow in development mode
    cd ../../
    pip install -e .
    cd examples/mongodb_integration/
    
    print_status "Dependencies installed"
}

# Function to start MongoDB using Docker
start_mongodb_docker() {
    print_info "Starting MongoDB using Docker Compose..."
    
    if ! command -v docker-compose >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Start MongoDB service only
    docker-compose up -d mongodb
    
    # Wait for MongoDB to be ready
    print_info "Waiting for MongoDB to be ready..."
    for i in {1..30}; do
        if docker-compose exec -T mongodb mongosh --eval "db.adminCommand('ismaster')" >/dev/null 2>&1; then
            print_status "MongoDB is ready"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    if [ $i -eq 30 ]; then
        print_error "MongoDB failed to start within 60 seconds"
        docker-compose logs mongodb
        exit 1
    fi
}

# Function to run the test
run_test() {
    local tracking_uri=${1:-"mongodb://localhost:27017/genesis_flow_test"}
    local artifact_root=${2:-"file:///tmp/genesis_flow_artifacts"}
    
    print_info "Running Genesis-Flow MongoDB integration test..."
    print_info "Tracking URI: $tracking_uri"
    print_info "Artifact Root: $artifact_root"
    
    export MLFLOW_TRACKING_URI="$tracking_uri"
    export MLFLOW_DEFAULT_ARTIFACT_ROOT="$artifact_root"
    
    # Create artifacts directory if using file storage
    if [[ $artifact_root == file://* ]]; then
        local local_path=${artifact_root#file://}
        mkdir -p "$local_path"
    fi
    
    # Run the test
    python test_genesis_flow_mongodb.py
    
    if [ $? -eq 0 ]; then
        print_status "Test completed successfully!"
    else
        print_error "Test failed!"
        exit 1
    fi
}

# Function to cleanup Docker resources
cleanup_docker() {
    print_info "Cleaning up Docker resources..."
    docker-compose down -v
    print_status "Cleanup completed"
}

# Function to show MongoDB data
show_mongodb_data() {
    local uri=${1:-"mongodb://localhost:27017/genesis_flow_test"}
    
    print_info "Showing MongoDB data..."
    
    if [[ $uri == *"mongodb"* ]]; then
        print_info "Starting mongo-express for data visualization..."
        docker-compose up -d mongo-express
        print_status "Mongo Express available at: http://localhost:8081"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  local           Test with local MongoDB (default)"
    echo "  docker          Test with MongoDB in Docker"
    echo "  docker-full     Test with Docker including visualization"
    echo "  cosmos          Test with Azure Cosmos DB (requires env vars)"
    echo "  install         Install dependencies only"
    echo "  cleanup         Cleanup Docker resources"
    echo "  show-data       Show MongoDB data in browser"
    echo "  help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  COSMOS_CONNECTION_STRING    Azure Cosmos DB connection string"
    echo "  AZURE_STORAGE_CONNECTION_STRING    Azure Blob Storage connection string"
    echo ""
    echo "Examples:"
    echo "  $0 local                    # Test with local MongoDB"
    echo "  $0 docker                   # Test with Docker MongoDB"
    echo "  $0 cosmos                   # Test with Azure Cosmos DB"
    echo ""
}

# Main script logic
case "${1:-local}" in
    "install")
        install_dependencies
        ;;
    
    "local")
        print_info "Testing with local MongoDB..."
        
        if check_mongodb; then
            print_status "Local MongoDB is running"
            install_dependencies
            run_test "mongodb://localhost:27017/genesis_flow_test"
        else
            print_warning "Local MongoDB is not running"
            print_info "Please start MongoDB or use 'docker' option"
            print_info "Install MongoDB: brew install mongodb/brew/mongodb-community"
            print_info "Start MongoDB: brew services start mongodb/brew/mongodb-community"
            exit 1
        fi
        ;;
    
    "docker")
        print_info "Testing with Docker MongoDB..."
        install_dependencies
        start_mongodb_docker
        run_test "mongodb://genesis:genesis_flow_test@localhost:27017/genesis_flow_test?authSource=admin"
        ;;
    
    "docker-full")
        print_info "Testing with Docker MongoDB and visualization..."
        install_dependencies
        start_mongodb_docker
        run_test "mongodb://genesis:genesis_flow_test@localhost:27017/genesis_flow_test?authSource=admin"
        show_mongodb_data
        ;;
    
    "cosmos")
        print_info "Testing with Azure Cosmos DB..."
        
        if [ -z "$COSMOS_CONNECTION_STRING" ]; then
            print_error "COSMOS_CONNECTION_STRING environment variable is required"
            print_info "Set it with: export COSMOS_CONNECTION_STRING='mongodb://...'"
            exit 1
        fi
        
        install_dependencies
        
        local artifact_root="file:///tmp/genesis_flow_artifacts"
        if [ -n "$AZURE_STORAGE_CONNECTION_STRING" ]; then
            artifact_root="azure://genesis-flow-artifacts/"
            print_info "Using Azure Blob Storage for artifacts"
        fi
        
        run_test "$COSMOS_CONNECTION_STRING" "$artifact_root"
        ;;
    
    "cleanup")
        cleanup_docker
        ;;
    
    "show-data")
        show_mongodb_data
        ;;
    
    "help")
        show_usage
        ;;
    
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac

print_status "Genesis-Flow MongoDB integration test runner completed!"