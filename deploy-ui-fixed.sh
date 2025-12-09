set -e  # Exit on any error

# Your Configuration
export PROJECT="mlops-476419"
export REGION="us-central1"
export IMAGE_UI="us-central1-docker.pkg.dev/${PROJECT}/onboarding-repo/onboarding-ui"

echo "========================================="
echo "🚀 Deploying Streamlit UI"
echo "========================================="
echo "Project: $PROJECT"
echo "Region: $REGION"
echo "Image: $IMAGE_UI:latest"
echo ""

# Step 1: Create the fixed Dockerfile with better requirements filtering
echo "📝 Creating Dockerfile.ui with Cloud Run optimizations..."
cat > Dockerfile.ui << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Copy entire repo to avoid missing files
COPY . .

# Filter bad requirements - including bias_detection and other invalid packages
RUN sed -i '/^__future__$/d; /^airflow$/d; /^Model_Pipeline$/d; /^bias_detection$/d; /^\s*$/d' Model_Pipeline/requirements.txt && \
    pip install --no-cache-dir -r Model_Pipeline/requirements.txt

EXPOSE 8080

# CRITICAL: Must set port and address for Cloud Run
ENTRYPOINT ["streamlit", "run", "/app/Model_Pipeline/app.py", \
            "--server.port=8080", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.serverAddress=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
EOF

echo "✅ Dockerfile.ui created"
echo ""

# Step 2: Create cloudbuild.yaml for custom Dockerfile
echo "📝 Creating cloudbuild.yaml..."
cat > cloudbuild.ui.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '${IMAGE_UI}:latest'
      - '-f'
      - 'Dockerfile.ui'
      - '.'
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - '${IMAGE_UI}:latest'
images:
  - '${IMAGE_UI}:latest'
timeout: 1200s
EOF

echo "✅ cloudbuild.ui.yaml created"
echo ""

# Step 3: Build & push using Cloud Build with custom config
echo "🔨 Building Docker image (this takes 5-10 minutes)..."
gcloud builds submit . \
  --config=cloudbuild.ui.yaml \
  --project="${PROJECT}" \
  --timeout=20m

echo "✅ Image built and pushed: ${IMAGE_UI}:latest"
echo ""

# Step 4: Get backend API URL automatically
echo "🔍 Getting backend API URL..."
API_URL=$(gcloud run services describe onboarding-api \
  --region=${REGION} \
  --project=${PROJECT} \
  --format='value(status.url)')

if [ -z "$API_URL" ]; then
  echo "❌ ERROR: Could not find backend API URL"
  echo "Make sure 'onboarding-api' service exists"
  exit 1
fi

echo "✅ Backend API found: $API_URL"
echo ""

# Step 5: Deploy to Cloud Run with correct settings
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy onboarding-ui \
  --image="${IMAGE_UI}:latest" \
  --region=${REGION} \
  --platform=managed \
  --project=${PROJECT} \
  --allow-unauthenticated \
  --set-env-vars="API_URL=${API_URL}" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --concurrency=50 \
  --port=8080

echo ""
echo "========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================="

# Get the service URL
SERVICE_URL=$(gcloud run services describe onboarding-ui \
  --region=${REGION} \
  --project=${PROJECT} \
  --format='value(status.url)')

echo ""
echo "🌐 Your Streamlit UI is live at:"
echo "   $SERVICE_URL"
echo ""
echo "🔗 Backend API connected to:"
echo "   $API_URL"
echo ""
echo "📊 View logs:"
echo "   gcloud run services logs read onboarding-ui --project=${PROJECT} --limit=50"
echo ""
echo "🔍 Check service status:"
echo "   gcloud run services describe onboarding-ui --region=${REGION} --project=${PROJECT}"
echo ""
echo "🧹 Cleanup build files:"
echo "   rm Dockerfile.ui cloudbuild.ui.yaml"
echo ""
echo "========================================="# [Copy the script content from the artifact above]
