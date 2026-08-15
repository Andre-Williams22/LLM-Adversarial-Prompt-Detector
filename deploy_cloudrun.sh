#!/usr/bin/env bash
#
# Deploy the adversarial prompt detector to Google Cloud Run.
#
# The recommended path is a Cloud Build trigger on push, which runs the same
# cloudbuild.yaml this script submits. Use this script for one-off deploys,
# for the first deploy before a trigger exists, or for inspecting a live
# service. Run "./deploy_cloudrun.sh help" for the full command list.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Defaults match cloudbuild.yaml so the two deployment paths cannot drift.
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-west1}"
SERVICE_NAME="${SERVICE_NAME:-adversarial-prompt-detector}"
REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/llm-adversarial-prompt-detector"
IMAGE_NAME="${REPOSITORY}/${SERVICE_NAME}"

check_prerequisites() {
    print_status "Checking prerequisites"

    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed"
        print_status "Install from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_error "Not logged into Google Cloud. Run: gcloud auth login"
        exit 1
    fi

    print_success "Prerequisites satisfied"
}

require_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. See https://docs.docker.com/get-docker/"
        exit 1
    fi
}

setup_project() {
    if [ -z "$PROJECT_ID" ]; then
        read -r -p "Google Cloud Project ID: " PROJECT_ID
        export PROJECT_ID
        REPOSITORY="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/llm-adversarial-prompt-detector"
        IMAGE_NAME="${REPOSITORY}/${SERVICE_NAME}"
    fi

    print_status "Configuring project ${PROJECT_ID}"
    gcloud config set project "$PROJECT_ID"

    print_status "Enabling required APIs"
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        artifactregistry.googleapis.com

    print_success "Project configured"
}

report_endpoints() {
    local url
    url=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" \
        --format="value(status.url)")

    print_success "Deployed to ${url}"
    echo ""
    echo "  Chat interface: ${url}/chat"
    echo "  Detect API:     ${url}/detect"
    echo "  Health:         ${url}/health"
    echo "  Readiness:      ${url}/ready"
    echo "  Metrics:        ${url}/metrics"
    echo "  API docs:       ${url}/docs"
    echo ""
}

deploy_with_cloud_build() {
    print_status "Submitting build to Cloud Build"
    gcloud builds submit \
        --config cloudbuild.yaml \
        --substitutions "_REGION=${REGION},_SERVICE_NAME=${SERVICE_NAME}" \
        .
    print_success "Cloud Build deployment complete"
    report_endpoints
}

deploy_manual() {
    require_docker

    print_status "Building image locally"
    docker build -t "${IMAGE_NAME}:latest" .

    print_status "Pushing to Artifact Registry"
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    docker push "${IMAGE_NAME}:latest"

    print_status "Deploying to Cloud Run"
    gcloud run deploy "$SERVICE_NAME" \
        --image "${IMAGE_NAME}:latest" \
        --platform managed \
        --region "$REGION" \
        --allow-unauthenticated \
        --port 80 \
        --memory 4Gi \
        --cpu 4 \
        --cpu-boost \
        --concurrency 100 \
        --min-instances 1 \
        --max-instances 12 \
        --timeout 3600 \
        --set-env-vars "TOKENIZERS_PARALLELISM=false,PYTORCH_ENABLE_MPS_FALLBACK=1,HF_HUB_DISABLE_SYMLINKS_WARNING=1,PORT=80"

    print_success "Manual deployment complete"
    report_endpoints
}

setup_monitoring() {
    print_status "Enabling Cloud Monitoring"
    gcloud services enable monitoring.googleapis.com
    print_success "Monitoring enabled; view metrics in the Cloud Console"
}

show_logs() {
    gcloud run services logs read "$SERVICE_NAME" --region="$REGION" --limit=50
}

check_status() {
    gcloud run services describe "$SERVICE_NAME" --region="$REGION"

    local url
    url=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" \
        --format="value(status.url)")

    print_status "Testing ${url}/health"
    curl -fsS "${url}/health" || print_error "Health check failed"
}

cleanup() {
    print_warning "This deletes the Cloud Run service and its images."
    read -r -p "Continue? (y/N): " confirm

    if [[ "$confirm" =~ ^[yY]([eE][sS])?$ ]]; then
        gcloud run services delete "$SERVICE_NAME" --region="$REGION" --quiet
        gcloud artifacts docker images delete "${IMAGE_NAME}" --quiet --delete-tags
        print_success "Cleanup complete"
    else
        print_status "Cleanup cancelled"
    fi
}

usage() {
    cat <<'USAGE'
Adversarial Prompt Detector - Cloud Run deployment

Usage: ./deploy_cloudrun.sh [COMMAND]

Commands:
  setup          Configure the project and enable required APIs
  deploy         Deploy via Cloud Build (recommended)
  deploy-manual  Build locally, push, and deploy with gcloud
  monitoring     Enable Cloud Monitoring
  logs           Tail recent service logs
  status         Describe the service and probe /health
  cleanup        Delete the service and its images
  all            setup + deploy + monitoring
  help           Show this message

Environment:
  PROJECT_ID     Google Cloud project (prompted if unset)
  REGION         Deployment region (default: us-west1)
  SERVICE_NAME   Cloud Run service name (default: adversarial-prompt-detector)

Examples:
  PROJECT_ID=my-project ./deploy_cloudrun.sh all
  ./deploy_cloudrun.sh status

For continuous deployment, point a Cloud Build trigger at cloudbuild.yaml
instead of running this script on every change.
USAGE
}

main() {
    case "${1:-help}" in
        setup)
            check_prerequisites
            setup_project
            ;;
        deploy)
            check_prerequisites
            setup_project
            deploy_with_cloud_build
            ;;
        deploy-manual)
            check_prerequisites
            setup_project
            deploy_manual
            ;;
        monitoring) setup_monitoring ;;
        logs)       show_logs ;;
        status)     check_status ;;
        cleanup)    cleanup ;;
        all)
            check_prerequisites
            setup_project
            deploy_with_cloud_build
            setup_monitoring
            print_success "Deployment finished"
            ;;
        help|*) usage ;;
    esac
}

main "$@"
