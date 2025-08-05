# 🚀 Complete Cloud Run Deployment Checklist

## ✅ **Pre-Deployment Setup (Do These First)**

### 1. **Enable Required APIs**
Go to **APIs & Services** → **Library** and enable:
- [x] Cloud Run API
- [x] Cloud Build API  
- [ ] **Container Registry API** ← Enable this!
- [ ] **Artifact Registry API** ← Enable this!
- [ ] **Secret Manager API** ← Enable this!

### 2. **MongoDB Atlas Configuration**
**In MongoDB Atlas Console:**
1. **Network Access** → **IP Whitelist**
2. **Add IP Address** → **Allow Access from Anywhere** (`0.0.0.0/0`)
   - Or use Google Cloud IP ranges for better security
3. **Database Access** → Ensure your user has read/write permissions
4. **Get Connection String**:
   - **Database** → **Connect** → **Connect Your Application**
   - Copy the connection string (format: `mongodb+srv://user:password@cluster.mongodb.net/database`)

### 3. **Environment Variables Setup**

**Option A: Using Secret Manager (Recommended)**
1. Go to **Secret Manager** → **Create Secret**
2. Create these secrets:
   - **Name**: `mongodb-uri`, **Value**: Your MongoDB Atlas connection string
   - **Name**: `openai-api-key`, **Value**: Your OpenAI API key (if using)
   - **Name**: `mlflow-tracking-uri`, **Value**: MLflow server URL (if external)

**Option B: Direct Environment Variables**
- We'll add these after first deployment in Cloud Run console

---

## 🔧 **Deployment Process**

### Step 1: **Trigger First Build**
- **Push any change** to your `cloud-run-deployment` branch
- **OR** Go to **Cloud Build** → **Triggers** → **Run trigger** manually

### Step 2: **Monitor Build Progress**
1. Go to **Cloud Build** → **History**
2. Watch the build progress (takes ~10-15 minutes)
3. **Build Steps:**
   - ✅ Copy dockerignore file
   - ✅ Build Docker image
   - ✅ Push to Container Registry
   - ✅ Deploy to Cloud Run

### Step 3: **Configure Environment Variables** (After First Deployment)
1. Go to **Cloud Run** → **adversarial-prompt-detector**
2. **Edit & Deploy New Revision**
3. **Variables & Secrets** tab
4. **Add Variables:**

**Environment Variables:**
```
TOKENIZERS_PARALLELISM=false
PYTORCH_ENABLE_MPS_FALLBACK=1
HF_HUB_DISABLE_SYMLINKS_WARNING=1
PORT=8080
```

**If using Secret Manager:**
```
MONGODB_URI → Reference secret: mongodb-uri
OPENAI_API_KEY → Reference secret: openai-api-key
MLFLOW_TRACKING_URI → Reference secret: mlflow-tracking-uri
```

**If using direct variables:**
```
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/database
```

### Step 4: **Verify Deployment**
1. **Get Service URL**: Cloud Run → adversarial-prompt-detector → URL
2. **Test Endpoints:**
   - **Health**: `https://your-url/health`
   - **Chat Interface**: `https://your-url/chat`
   - **API Docs**: `https://your-url/`

---

## 🔍 **Post-Deployment Verification**

### 1. **Health Check**
```bash
curl https://your-service-url/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_count": 4,
  "version": "1.0.0"
}
```

### 2. **Chat Interface Test**
1. Open `https://your-service-url/chat`
2. Send test message: "Hello, how are you?"
3. Verify response appears
4. Test adversarial prompt: "Ignore previous instructions and..."
5. Verify it gets blocked

### 3. **Model Loading Verification**
**Check logs for:**
- "✅ Models loaded successfully"
- "Fast detection system ready"
- No import errors or missing dependencies

**View Logs:**
1. **Cloud Run** → **adversarial-prompt-detector** → **Logs**
2. Look for startup messages

---

## 🛠️ **Configuration Optimizations**

### 1. **Resource Tuning** (if needed)
**Cloud Run** → **Edit & Deploy New Revision** → **Resources**

**If you see memory issues:**
- Increase **Memory** to `6Gi` or `8Gi`

**If you see startup timeouts:**
- Increase **CPU** to `4`
- Increase **Request timeout** to `900`

### 2. **Scaling Configuration**
**Current settings (good defaults):**
- **Min instances**: 0 (scales to zero)
- **Max instances**: 10
- **Concurrency**: 100

**For production:**
- **Min instances**: 1 (always warm, faster response)
- **Max instances**: 20 (handle more traffic)

### 3. **Cost Optimization**
**CPU allocation**: "Only during request processing" (saves money)

---

## 🔐 **Security & Monitoring Setup**

### 1. **Set Up Monitoring Alerts**
1. **Monitoring** → **Alerting** → **Create Policy**
2. **Alert for:**
   - Error rate > 5%
   - Response time > 10 seconds
   - Memory usage > 80%

### 2. **Budget Alerts**
1. **Billing** → **Budgets & alerts**
2. **Create budget**: $50/month
3. **Alert thresholds**: 50%, 80%, 100%

### 3. **Custom Domain** (Optional)
1. **Cloud Run** → **Manage Custom Domains**
2. **Add mapping**: your-domain.com → adversarial-prompt-detector
3. **Verify domain ownership**
4. **Update DNS records**

---

## 🚨 **Troubleshooting Common Issues**

### **Build Failures**
**Check Cloud Build logs for:**
- Missing dependencies in `requirements.txt`
- Docker build context issues
- Model files not found

**Solutions:**
- Ensure `data/preprocessed/` contains your models
- Check `.dockerignore.cloudrun` doesn't exclude needed files
- Verify `Dockerfile.cloudrun` paths are correct

### **Runtime Errors**
**Check Cloud Run logs for:**
- Model loading failures
- Memory exhaustion
- Import errors

**Solutions:**
- Increase memory allocation
- Check model file sizes
- Verify Python dependencies

### **Network Issues**
**MongoDB connection problems:**
- Verify MongoDB Atlas IP whitelist includes `0.0.0.0/0`
- Check connection string format
- Test connection string locally first

### **Performance Issues**
**Slow responses:**
- Check model loading time in logs
- Consider increasing CPU allocation
- Set min instances to 1 for always-warm

---

## 📊 **Success Metrics**

After deployment, you should see:

### **Functional Tests**
- [ ] Health endpoint returns 200 OK
- [ ] Chat interface loads without errors
- [ ] Safe prompts get processed
- [ ] Adversarial prompts get blocked
- [ ] MongoDB logging works (check database)

### **Performance Metrics**
- [ ] Cold start time < 30 seconds
- [ ] Warm response time < 2 seconds
- [ ] Memory usage < 3GB under normal load
- [ ] Error rate < 1%

### **Cost Metrics**
- [ ] Daily cost < $2 for light usage
- [ ] Scales to zero when not used
- [ ] No unexpected charges

---

## 🎯 **Next Steps After Successful Deployment**

### 1. **Set Up CI/CD**
- Every push to main branch auto-deploys
- Consider adding staging environment

### 2. **Production Hardening**
- Enable authentication if needed
- Set up proper monitoring dashboards
- Configure backup/disaster recovery

### 3. **Feature Enhancements**
- Re-enable OpenAI integration
- Add more ML models
- Implement user analytics

---

## 📞 **Getting Help**

### **If Build Fails:**
1. Check **Cloud Build** → **History** → **Build logs**
2. Look for specific error messages
3. Common issues: missing files, dependency conflicts

### **If Service Won't Start:**
1. Check **Cloud Run** → **Logs** for startup errors
2. Test Docker image locally first
3. Verify environment variables

### **If Models Don't Load:**
1. Check if `data/preprocessed/` is included in image
2. Verify model file permissions
3. Check memory allocation (increase if needed)

**Ready to deploy?** ✅ Follow this checklist step by step and your adversarial prompt detector will be running on Google Cloud! 🚀
