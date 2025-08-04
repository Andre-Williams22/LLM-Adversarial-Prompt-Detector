#!/bin/bash

# Google Cloud Run Deployment Script
# Automated deployment for Adversarial Prompt Detector

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC=    echo ""
    echo "🛡️ Adversarial Prompt Detector - Google Cloud Run Deployment"
    echo "=============================================================="
    echo ""
    echo "📋 DEPLOYMENT OPTIONS:"
    echo ""
    echo "🌟 OPTION A: GitHub Integration (RECOMMENDED - No CLI needed!)"
    echo "   • Go to: https://console.cloud.google.com"
    echo "   • Cloud Build → Triggers → Create Trigger"
    echo "   • Connect your GitHub repo"
    echo "   • Use cloudbuild.yaml configuration"
    echo "   • Push to deploy automatically!"
    echo ""
    echo "🔧 OPTION B: CLI Deployment (what this script does)"
    echo "   • Requires Google Cloud SDK installed"
    echo "   • Manual control over deployment process"
    echo "   • Good for testing and development"
    echo ""3[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="adversarial-prompt-detector"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed!"
        print_status "Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        print_status "Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check if logged into gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_error "Not logged into Google Cloud!"
        print_status "Run: gcloud auth login"
        exit 1
    fi
    
    print_success "Prerequisites check passed!"
}

# Function to setup project
setup_project() {
    if [ -z "$PROJECT_ID" ]; then
        print_status "No PROJECT_ID set. Please provide your Google Cloud Project ID:"
        read -p "Project ID: " PROJECT_ID
        export PROJECT_ID
    fi
    
    print_status "Setting up Google Cloud project: $PROJECT_ID"
    
    # Set the project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_status "Enabling required Google Cloud APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com
    
    print_success "Project setup complete!"
}

# Function to build and deploy using Cloud Build
deploy_with_cloud_build() {
    print_status "🚀 Deploying with Cloud Build..."
    
    # Submit build to Cloud Build
    gcloud builds submit \
        --config cloudbuild.yaml \
        --substitutions _REGION=$REGION,_SERVICE_NAME=$SERVICE_NAME \
        .
    
    if [ $? -eq 0 ]; then
        print_success "✅ Cloud Build deployment successful!"
        
        # Get the service URL
        SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
        
        print_success "🌐 Your application is deployed!"
        echo ""
        echo "📱 Chat Interface: $SERVICE_URL/chat"
        echo "🔍 Health Check: $SERVICE_URL/health"
        echo "📊 Metrics: $SERVICE_URL/metrics"
        echo "📖 API Docs: $SERVICE_URL"
        echo ""
        
    else
        print_error "❌ Cloud Build deployment failed!"
        exit 1
    fi
}

# Function to build and deploy manually
deploy_manual() {
    print_status "🔨 Building Docker image locally..."
    
    # Build the image
    docker build -f Dockerfile.cloudrun -t $IMAGE_NAME:latest .
    
    print_status "📤 Pushing image to Container Registry..."
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker
    
    # Push the image
    docker push $IMAGE_NAME:latest
    
    print_status "🚀 Deploying to Cloud Run..."
    
    # Deploy to Cloud Run
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --concurrency 100 \
        --max-instances 10 \
        --min-instances 0 \
        --timeout 300 \
        --port 8080 \
        --set-env-vars "TOKENIZERS_PARALLELISM=false,PYTORCH_ENABLE_MPS_FALLBACK=1,HF_HUB_DISABLE_SYMLINKS_WARNING=1"
    
    if [ $? -eq 0 ]; then
        print_success "✅ Manual deployment successful!"
        
        # Get the service URL
        SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
        
        print_success "🌐 Your application is deployed!"
        echo ""
        echo "📱 Chat Interface: $SERVICE_URL/chat"
        echo "🔍 Health Check: $SERVICE_URL/health"
        echo "📊 Metrics: $SERVICE_URL/metrics"
        echo "📖 API Docs: $SERVICE_URL"
        echo ""
        
    else
        print_error "❌ Manual deployment failed!"
        exit 1
    fi
}

# Function to setup monitoring
setup_monitoring() {
    print_status "Setting up Cloud Monitoring..."
    
    # Enable monitoring API
    gcloud services enable monitoring.googleapis.com
    
    print_status "Monitoring setup complete. Check Google Cloud Console for metrics."
}

# Function to show logs
show_logs() {
    print_status "📋 Recent application logs:"
    gcloud run services logs read $SERVICE_NAME --region=$REGION --limit=50
}

# Function to check service status
check_status() {
    print_status "Checking service status..."
    
    # Get service info
    gcloud run services describe $SERVICE_NAME --region=$REGION
    
    # Test health endpoint
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    
    print_status "Testing health endpoint..."
    curl -s "$SERVICE_URL/health" | jq . || echo "Health check failed"
}

# Function to cleanup resources
cleanup() {
    print_warning "⚠️  This will delete your Cloud Run service and images!"
    read -p "Are you sure? (y/N): " confirm
    
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        print_status "Cleaning up resources..."
        
        # Delete Cloud Run service
        gcloud run services delete $SERVICE_NAME --region=$REGION --quiet
        
        # Delete images
        gcloud container images delete $IMAGE_NAME:latest --quiet
        
        print_success "Cleanup complete!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Main function
main() {
    echo ""
    echo "🛡️  Adversarial Prompt Detector - Google Cloud Run Deployment"
    echo "=============================================================="
    echo ""
    
    case "${1:-help}" in
        "setup")
            check_prerequisites
            setup_project
            ;;
        "deploy")
            check_prerequisites
            setup_project
            deploy_with_cloud_build
            ;;
        "deploy-manual")
            check_prerequisites
            setup_project
            deploy_manual
            ;;
        "monitoring")
            setup_monitoring
            ;;
        "logs")
            show_logs
            ;;
        "status")
            check_status
            ;;
        "cleanup")
            cleanup
            ;;
        "all")
            check_prerequisites
            setup_project
            deploy_with_cloud_build
            setup_monitoring
            print_success "🎉 Complete deployment finished!"
            ;;
        "help"|*)
            echo ""
            echo "🛡️ Adversarial Prompt Detector - Google Cloud Run Deployment"
            echo ""
            echo "🌟 EASIEST WAY (No CLI needed):"
            echo "   1. Go to: https://console.cloud.google.com"
            echo "   2. Cloud Build → Triggers → Create Trigger"
            echo "   3. Connect your GitHub repository"
            echo "   4. Use 'cloudbuild.yaml' as configuration"
            echo "   5. Push code to auto-deploy!"
            echo ""
            echo "🔧 CLI DEPLOYMENT (this script):"
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  setup         Setup Google Cloud project and APIs"
            echo "  deploy        Deploy using Cloud Build (recommended)"
            echo "  deploy-manual Deploy manually with local Docker build"
            echo "  monitoring    Setup Cloud Monitoring"
            echo "  logs          Show recent application logs"
            echo "  status        Check service status and test endpoints"
            echo "  cleanup       Delete Cloud Run service and images"
            echo "  all           Complete setup and deployment"
            echo "  help          Show this help"
            echo ""
            echo "Examples:"
            echo "  PROJECT_ID=my-project $0 all        # Complete deployment"
            echo "  $0 deploy                           # Deploy only"
            echo "  $0 logs                             # View logs"
            echo "  $0 status                           # Check status"
            echo ""
            echo "Prerequisites for CLI:"
            echo "  - Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
            echo "  - Docker (for manual deployment): https://docs.docker.com/get-docker/"
            echo "  - PROJECT_ID environment variable or interactive input"
            echo ""
            echo "💡 TIP: Use GitHub integration for the easiest experience!"
            echo ""
            ;;
    esac
}

# Run main with all arguments
main "$@"
