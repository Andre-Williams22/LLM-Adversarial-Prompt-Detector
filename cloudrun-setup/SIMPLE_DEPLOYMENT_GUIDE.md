# 🚀 Simple Cloud Run Deployment - No CLI Required!

## ✅ What You Need (Prerequisites)
- ✅ Google Cloud Project with PROJECT_ID
- ✅ Cloud Run API enabled
- ✅ Cloud Build API enabled
- ✅ This code repository

## 🎯 Two Deployment Options

### Option A: GitHub Integration (Recommended - No CLI needed!)
### Option B: Manual Upload (Requires CLI)

---

## 🌟 Option A: GitHub Integration (EASIEST - No CLI Required!)

This uses GitHub + Cloud Build trigger for completely automated deployments.

### Step 1: Connect Your Repository to Cloud Build

1. **Go to Google Cloud Console**: https://console.cloud.google.com
2. **Select your project** (use your PROJECT_ID)
3. **Navigate to Cloud Build** → **Triggers**
4. **Click "Create Trigger"**

### Step 2: Configure the Trigger

Fill in these details:

**Name**: `adversarial-prompt-detector-deploy`

**Event**: `Push to a branch`

**Source**:
- **Repository**: Connect your GitHub repository
- **Branch**: `^main$` (or your default branch)

**Configuration**:
- **Type**: `Cloud Build configuration file (yaml or json)`
- **Location**: `cloudbuild.yaml`

**Advanced** (click to expand):
- **Substitution variables**: Leave empty (we use defaults)

### Step 3: Deploy!

1. **Click "Create"** - This creates the trigger
2. **Push any change** to your main branch, or
3. **Click "Run trigger"** manually in the Cloud Build console

### Step 4: Check Your Deployment

1. **Go to Cloud Run** in the console
2. **Find your service**: `adversarial-prompt-detector`
3. **Click the URL** - Your app is live! 🎉

**Your URLs will be**:
- **Chat Interface**: `https://your-service-url/chat`
- **Health Check**: `https://your-service-url/health`
- **API Docs**: `https://your-service-url/`

---

## 🔧 Option B: Manual Upload (Requires CLI)

If you prefer manual control or want to test locally first.

### Step 1: Install Google Cloud CLI

**Mac/Linux**:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Windows**: Download from https://cloud.google.com/sdk/docs/install

### Step 2: Authenticate and Set Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Verify setup
gcloud config list
```

### Step 3: Deploy with One Command

```bash
# Navigate to your project directory
cd /path/to/your/LLM-Adversarial-Prompt-Detector

# Deploy everything
./deploy_cloudrun.sh deploy
```

**Alternative - Manual build submission**:
```bash
gcloud builds submit --config cloudbuild.yaml
```

---

## 🎛️ GCP Console Configuration (Optional)

You **don't need** to do anything in the GCP UI for basic deployment, but here are useful things you can configure:

### 1. **Cloud Run Service Settings** (Optional)
Go to **Cloud Run** → **adversarial-prompt-detector** → **Edit & Deploy New Revision**

**Useful settings to adjust**:
- **CPU allocation**: Only during request processing (saves money)
- **Memory**: Increase to 6Gi if you see memory errors
- **Max instances**: Increase to 20 for high traffic
- **Min instances**: Set to 1 if you need faster response (costs more)

### 2. **Monitoring & Alerting** (Recommended)
Go to **Monitoring** → **Alerting**

**Create alerts for**:
- **High error rate**: >5% errors
- **High latency**: >10 seconds response time
- **Cost alerts**: Set budget alerts

### 3. **Custom Domain** (Optional)
Go to **Cloud Run** → **Manage Custom Domains**

**Steps**:
1. Add your domain
2. Verify ownership
3. Update DNS records

---

## 🔍 Troubleshooting

### Build Failures

**Check build logs**:
1. Go to **Cloud Build** → **History**
2. Click on failed build
3. Check logs for errors

**Common issues**:
- **Dockerfile path**: Make sure `Dockerfile.cloudrun` exists
- **Requirements**: Ensure `requirements.txt` is complete
- **Memory**: Build might need more memory (increase in cloudbuild.yaml)

### Runtime Errors

**Check application logs**:
1. Go to **Cloud Run** → **adversarial-prompt-detector** → **Logs**
2. Look for startup errors

**Common issues**:
- **Model loading**: Check if models are in `data/preprocessed/`
- **Memory**: Increase service memory to 6Gi or 8Gi
- **Port**: Make sure app listens on PORT environment variable

### Service Not Starting

**Check health endpoint**:
```bash
curl https://your-service-url/health
```

**If unhealthy**:
1. Check logs for Python errors
2. Verify all dependencies in requirements.txt
3. Test locally with Docker first

---

## 💰 Cost Monitoring

### Set Budget Alerts
1. Go to **Billing** → **Budgets & alerts**
2. Create budget for $25/month
3. Set alerts at 50%, 80%, 100%

### Monitor Usage
- **Cloud Run**: Check requests and resource usage
- **Container Registry**: Storage costs for images
- **Cloud Build**: Build minutes used

**Expected costs**:
- **Light usage**: $3-10/month
- **Medium usage**: $15-30/month
- **Heavy usage**: $40-80/month

---

## 🎉 Success Checklist

After deployment, verify:

- [ ] **Service running**: Cloud Run shows "Receiving traffic"
- [ ] **Health check**: `/health` returns `{"status": "healthy"}`
- [ ] **Chat interface**: `/chat` loads Gradio interface
- [ ] **API working**: Can send requests to endpoints
- [ ] **Models loaded**: Check logs for "Models loaded successfully"
- [ ] **Monitoring**: Can see metrics in Cloud Console

---

## 🚀 What Happens During Deployment

### Cloud Build Process:
1. **Trigger activated** (GitHub push or manual)
2. **Dockerfile.cloudrun built** (installs dependencies, copies code)
3. **Image pushed** to Container Registry
4. **Cloud Run service updated** with new image
5. **Traffic routes** to new version automatically

### Resource Allocation:
- **CPU**: 2 cores (sufficient for ML inference)
- **Memory**: 4GB (handles your 4 detection models)
- **Concurrency**: 100 requests per instance
- **Scaling**: 0-10 instances (auto-scales based on traffic)

---

## 🔄 Future Updates

### Automatic Updates (GitHub Integration)
- **Push code** → **Auto-deploys** to Cloud Run
- **Zero downtime** deployments
- **Rollback available** if issues occur

### Manual Updates
```bash
# Update and redeploy
git push origin main
# OR
gcloud builds submit --config cloudbuild.yaml
```

**That's it!** Your adversarial prompt detector is now running on Google Cloud with automatic scaling and minimal costs! 🎉

**Need help?** Check the logs in Cloud Console or run the health check endpoint.
