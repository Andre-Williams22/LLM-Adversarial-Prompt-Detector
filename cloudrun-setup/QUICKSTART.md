# ⚡ QUICK START - No CLI Required!

## ✅ You Have:
- Google Cloud Project ID
- Cloud Run API enabled  
- Cloud Build API enabled
- This GitHub repository

## 🚀 Deploy in 3 Steps (Web UI Only):

### Step 1: Open Google Cloud Console
Go to: https://console.cloud.google.com
Select your project

### Step 2: Create Build Trigger
1. **Navigation**: Cloud Build → Triggers
2. **Click**: "Create Trigger"
3. **Fill in**:
   - **Name**: `deploy-adversarial-detector`
   - **Event**: Push to a branch
   - **Source**: Connect your GitHub repo
   - **Branch**: `^main$`
   - **Configuration**: Cloud Build configuration file
   - **Location**: `cloudbuild.yaml`
4. **Click**: "Create"

### Step 3: Deploy
**Option A**: Push any change to your main branch
**Option B**: Click "Run trigger" in Cloud Build console

## 🎉 Done!
Your app will be live at:
- **Service URL**: Check Cloud Run console
- **Chat**: `your-url/chat`
- **Health**: `your-url/health`

## 📊 Check Deployment Status:
1. **Cloud Build** → History (see build progress)
2. **Cloud Run** → Services (see running service)

---

## 🔧 Alternative: CLI Method

If you prefer command line:

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Login and deploy
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
./deploy_cloudrun.sh all
```

---

## 💰 Expected Costs:
- **Development**: $0-5/month (free tier)
- **Light production**: $5-15/month
- **Heavy usage**: $20-50/month

**Much cheaper than alternatives!** 🎯
