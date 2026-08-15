# Deploying LLM Adversarial Prompt Detector on CapRover with Google Compute Engine

*A complete guide to setting up a production-ready ML application with monitoring on Google Cloud Platform*

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Google Compute Engine Setup](#google-compute-engine-setup)
4. [CapRover Installation](#caprover-installation)
5. [Domain Configuration](#domain-configuration)
6. [Application Deployment](#application-deployment)
7. [Monitoring Stack Setup](#monitoring-stack-setup)
8. [Security Configuration](#security-configuration)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide walks you through deploying the LLM Adversarial Prompt Detector on Google Compute Engine using CapRover as the Platform-as-a-Service (PaaS) layer. CapRover provides Docker-based deployments with automatic SSL, load balancing, and easy application management.

### Architecture Overview

```
Internet → Domain/SSL → CapRover → Docker Containers
                                 ├── Main App (FastAPI + Gradio)
                                 ├── Prometheus (Metrics)
                                 ├── Grafana (Dashboards)
                                 └── MongoDB (Optional - can use Atlas)
```

### Why This Stack?

- **Cost-Effective**: GCE provides flexible pricing with sustained use discounts
- **Scalable**: Easy to upgrade VM resources or add load balancing
- **Production-Ready**: Automatic SSL, monitoring, and professional deployment
- **ML-Optimized**: Sufficient compute power for transformer models
- **Easy Management**: CapRover's web interface simplifies deployments

---

## Prerequisites

Before starting, ensure you have:

- **Google Cloud Platform account** with billing enabled
- **Domain name** (required for SSL certificates)
- **SSH key pair** for secure server access
- **Git repository** with the application code
- **Basic familiarity** with command line and Docker concepts

### Required Tools

Install these tools on your local machine:

```bash
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Optional: Verify installation
gcloud --version
```

---

## Google Compute Engine Setup

### Step 1: Create VM Instance

#### Option A: Using Google Cloud Console

1. **Navigate to Compute Engine** → **VM instances**
2. **Click "Create Instance"**
3. **Configure the instance:**

   ```
   Name: caprover-server
   Region: us-central1 (or closest to your users)
   Zone: us-central1-a

   Machine Configuration:
   - Series: E2 (cost-effective) or N2 (better performance)
   - Machine type: e2-standard-4 (4 vCPUs, 16 GB memory)

   Boot disk:
   - Operating system: Ubuntu
   - Version: Ubuntu 20.04 LTS
   - Boot disk type: SSD persistent disk
   - Size: 50 GB (minimum for ML models)

   Firewall:
   Allow HTTP traffic
   Allow HTTPS traffic
   ```

4. **Click "Create"**

#### Option B: Using gcloud CLI

```bash
# Set project and region
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Create the VM instance
gcloud compute instances create caprover-server \
    --machine-type=e2-standard-4 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server,https-server

# Get the external IP
gcloud compute instances describe caprover-server \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Step 2: Configure Firewall Rules

CapRover needs several ports open:

```bash
# Create firewall rules for CapRover
gcloud compute firewall-rules create caprover-http \
    --allow tcp:80 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow HTTP traffic to CapRover"

gcloud compute firewall-rules create caprover-https \
    --allow tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow HTTPS traffic to CapRover"

gcloud compute firewall-rules create caprover-captain \
    --allow tcp:3000 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow CapRover dashboard access"

# Optional: Restrict CapRover dashboard to your IP
# Replace YOUR_IP with your actual IP address
gcloud compute firewall-rules create caprover-captain-restricted \
    --allow tcp:3000 \
    --source-ranges YOUR_IP/32 \
    --description "Restricted CapRover dashboard access"
```

### Step 3: SSH Key Setup

```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# Add SSH key to the VM
gcloud compute ssh caprover-server --ssh-key-file=~/.ssh/id_rsa
```

---

## CapRover Installation

### Step 1: Connect to Server

```bash
# Connect via SSH
gcloud compute ssh caprover-server

# Or use direct SSH with external IP
ssh -i ~/.ssh/id_rsa username@EXTERNAL_IP
```

### Step 2: Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker GPG key and repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group
sudo usermod -aG docker $USER

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
```

### Step 3: Install CapRover

```bash
# Install CapRover
sudo npm install -g caprover

# Initialize CapRover
sudo caprover serversetup
```

**During CapRover setup, you'll be prompted for:**

```
? Have you already created a CapRover instance on a server? No
? IP address of your server: [ENTER YOUR SERVER'S EXTERNAL IP]
? CapRover server root domain: captain.yourdomain.com
? New CapRover password: [CREATE A STRONG PASSWORD]
? Email address for Let's Encrypt: your-email@example.com
```

### Step 4: Verify CapRover Installation

1. **Access CapRover dashboard:** `https://captain.yourdomain.com`
2. **Login with the password** you created
3. **Verify the dashboard loads** correctly

---

## Domain Configuration

### Step 1: DNS Setup

Configure your domain's DNS records:

```
# Main CapRover domain
captain.yourdomain.com → A → SERVER_EXTERNAL_IP

# Wildcard for applications
*.yourdomain.com → A → SERVER_EXTERNAL_IP

# Specific app subdomains (optional)
app.yourdomain.com → A → SERVER_EXTERNAL_IP
grafana.yourdomain.com → A → SERVER_EXTERNAL_IP
prometheus.yourdomain.com → A → SERVER_EXTERNAL_IP
```

### Step 2: SSL Certificate Setup

CapRover automatically handles SSL certificates via Let's Encrypt:

1. **Navigate to Settings** in CapRover dashboard
2. **Enable HTTPS** for your domain
3. **Force HTTPS** for security
4. **Verify SSL** certificate is issued

---

## Application Deployment

### Step 1: Prepare Application Files

First, ensure your project has the necessary deployment files:

```bash
# Clone your repository
git clone https://github.com/Andre-Williams22/LLM-Adversarial-Prompt-Detector.git
cd LLM-Adversarial-Prompt-Detector

# Verify deployment files exist
ls -la captain-definition
ls -la Dockerfile
ls -la requirements.txt
```

### Step 2: Create Main Application

1. **Access CapRover dashboard**
2. **Go to "Apps"** → **"Create New App"**
3. **Configure the app:**

   ```
   App Name: adversarial-detector
   Has persistent data
   Instance Count: 1
   ```

4. **Deploy the application:**

   **Option A: Git Repository**
   ```
   Repository: https://github.com/Andre-Williams22/LLM-Adversarial-Prompt-Detector.git
   Branch: main
   Username: [your-github-username]
   Password: [your-github-token]
   ```

   **Option B: Upload TAR file**
   ```bash
   # Create deployment package
   tar -czf app.tar.gz --exclude-vcs-ignores .
   ```
   Upload `app.tar.gz` via CapRover interface

### Step 3: Configure Environment Variables

In the CapRover app settings, add these environment variables:

```bash
# Core application settings
PORT=80
LOG_LEVEL=INFO

# Detection policy: high, balanced, or conservative
FAST_DETECTION_SENSITIVITY=balanced
EAGER_MODEL_LOAD=false

# Interaction logging (optional)
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/adversarial_detection
MONGODB_DATABASE=adversarial_detection

# Experiment tracking (optional)
MLFLOW_TRACKING_URI=http://mlflow.yourdomain.com
```

The full list of supported variables is in `.env.example`. Every one is
optional: with none set, the service still loads its models and classifies
prompts, and only the optional integrations are disabled.

### Step 4: Configure Health Check

```json
{
  "healthCheckPath": "/health",
  "containerHttpPort": 80,
  "forceSsl": true
}
```

### Step 5: Enable Custom Domain

1. **Go to HTTP Settings** in the app
2. **Add custom domain:** `app.yourdomain.com`
3. **Enable SSL** for the domain
4. **Verify the app** is accessible

---

## Monitoring Stack Setup

### Step 1: Deploy Prometheus

1. **Create new app:** `prometheus`
2. **Upload deployment package:**

   ```bash
   # Navigate to prometheus deployment
   cd deployments/prometheus-deployment

   # Create TAR package
   tar -czf prometheus-deployment.tar.gz *
   ```

3. **Configure environment variables:**
   ```bash
   RETENTION_TIME=15d
   STORAGE_PATH=/prometheus-data
   CONFIG_FILE=/etc/prometheus/prometheus.yml
   ```

4. **Set custom domain:** `prometheus.yourdomain.com`

### Step 2: Deploy Grafana

1. **Create new app:** `grafana`
2. **Upload deployment package:**

   ```bash
   # Navigate to grafana deployment
   cd deployments/grafana-with-dashboard

   # Create TAR package
   tar -czf grafana-deployment.tar.gz *
   ```

3. **Configure environment variables:**
   ```bash
   # Security (IMPORTANT: Change these!)
   GF_SECURITY_ADMIN_USER=admin
   GF_SECURITY_ADMIN_PASSWORD=your_secure_password_here

   # Data source
   GF_DATABASE_TYPE=sqlite3
   GF_DATABASE_PATH=/var/lib/grafana/grafana.db

   # Server settings
   GF_SERVER_ROOT_URL=https://grafana.yourdomain.com
   GF_SERVER_SERVE_FROM_SUB_PATH=false

   # Authentication
   GF_AUTH_ANONYMOUS_ENABLED=false
   GF_USERS_ALLOW_SIGN_UP=false
   ```

4. **Set custom domain:** `grafana.yourdomain.com`

### Step 3: Configure Data Sources

After Grafana deployment:

1. **Access Grafana:** `https://grafana.yourdomain.com`
2. **Login** with your admin credentials
3. **Add Prometheus data source:**
   ```
   URL: http://prometheus.yourdomain.com
   Access: Server (default)
   HTTP Method: GET
   ```

4. **Test connection** and save

### Step 4: Import Dashboards

Upload the pre-configured dashboard:

1. **Go to Dashboards** → **Import**
2. **Upload** `deployments/basic-dashboard.json`
3. **Select Prometheus** as the data source
4. **Save the dashboard**

---

## Security Configuration

### Step 1: Change Default Passwords

**Grafana Admin Password:**
```bash
# Via Grafana CLI (if accessing container)
docker exec -it cap-grafana grafana-cli admin reset-admin-password your_new_password

# Or update environment variables in CapRover
GF_SECURITY_ADMIN_PASSWORD=your_new_secure_password
```

**CapRover Admin Password:**
1. **Go to Settings** in CapRover dashboard
2. **Change Password** section
3. **Enter new password** and save

### Step 2: Configure Firewall

```bash
# Restrict CapRover dashboard access to your IP
gcloud compute firewall-rules update caprover-captain-restricted \
    --source-ranges YOUR_IP/32

# Optional: Block direct access to app ports
gcloud compute firewall-rules create block-direct-access \
    --deny tcp:8080,5000,9090 \
    --source-ranges 0.0.0.0/0 \
    --description "Block direct access to application ports"
```

### Step 3: Enable 2FA (Recommended)

For CapRover dashboard:
1. **Install authenticator app** (Google Authenticator, Authy)
2. **Go to Settings** → **Two Factor Authentication**
3. **Scan QR code** and enable 2FA

### Step 4: SSL/TLS Hardening

Ensure all services use HTTPS:

```bash
# Verify SSL certificates
curl -I https://app.yourdomain.com
curl -I https://grafana.yourdomain.com
curl -I https://prometheus.yourdomain.com

# Check SSL grade
# Visit: https://www.ssllabs.com/ssltest/
```

---

## Performance Optimization

### Step 1: Resource Allocation

**Main Application:**
```
CPU: 2000m (2 cores)
Memory: 4096Mi (4GB)
```

**Prometheus:**
```
CPU: 500m (0.5 cores)
Memory: 2048Mi (2GB)
```

**Grafana:**
```
CPU: 200m (0.2 cores)
Memory: 512Mi (512MB)
```

### Step 2: Enable Caching

Add to main application environment:

```bash
# Redis caching (if using Redis)
REDIS_URL=redis://redis.yourdomain.com:6379

# Application-level caching
CACHE_TTL=3600
CACHE_SIZE=100
```

### Step 3: Database Optimization

For MongoDB Atlas:
- **Enable connection pooling**
- **Set appropriate indexes**
- **Configure read preferences**

```bash
# MongoDB connection with optimizations
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/database?retryWrites=true&w=majority&maxPoolSize=10
```

### Step 4: Load Balancing (Optional)

For high traffic, configure Google Cloud Load Balancer:

```bash
# Create instance group
gcloud compute instance-groups unmanaged create caprover-group \
    --zone=us-central1-a

# Add instance to group
gcloud compute instance-groups unmanaged add-instances caprover-group \
    --instances=caprover-server \
    --zone=us-central1-a

# Create health check
gcloud compute health-checks create http caprover-health-check \
    --port=80 \
    --request-path=/health

# Create backend service
gcloud compute backend-services create caprover-backend \
    --health-checks=caprover-health-check \
    --global

# Add backend
gcloud compute backend-services add-backend caprover-backend \
    --instance-group=caprover-group \
    --instance-group-zone=us-central1-a \
    --global
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Application Won't Start

**Symptoms:** App shows as "failed" in CapRover
**Solutions:**
```bash
# Check application logs
# In CapRover: Apps → [app-name] → Logs

# Verify Docker image builds correctly
docker build -t test-app .
docker run -p 8080:80 test-app

# Check resource allocation
# Increase memory if models are large
```

#### 2. SSL Certificate Issues

**Symptoms:** Browser shows "Not Secure" or certificate errors
**Solutions:**
```bash
# Verify DNS propagation
nslookup app.yourdomain.com

# Check CapRover SSL settings
# Settings → HTTPS → Force HTTPS

# Manually trigger certificate renewal
# Settings → HTTPS → Re-enable HTTPS
```

#### 3. High Memory Usage

**Symptoms:** Applications crashing or slow performance
**Solutions:**
```bash
# Monitor resource usage
htop
docker stats

# Increase VM memory or optimize application
# Consider using model quantization for ML models

# Enable swap if needed (temporary solution)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Database Connection Issues

**Symptoms:** Application can't connect to MongoDB
**Solutions:**
```bash
# Verify connection string
echo $MONGODB_URI

# Test connection from server
curl -X POST http://localhost/api/test-db

# Check MongoDB Atlas IP whitelist
# Add 0.0.0.0/0 or specific server IP
```

#### 5. Prometheus Not Collecting Metrics

**Symptoms:** Empty Grafana dashboards
**Solutions:**
```bash
# Check Prometheus targets
curl http://prometheus.yourdomain.com/targets

# Verify application exposes /metrics endpoint
curl http://app.yourdomain.com/metrics

# Check Prometheus configuration
# Apps → prometheus → App Configs → prometheus.yml
```

### Log Analysis

**View application logs:**
```bash
# CapRover dashboard method
Apps → [app-name] → Logs

# Direct Docker method (SSH to server)
docker logs cap-adversarial-detector
docker logs cap-prometheus
docker logs cap-grafana
```

**Monitor system resources:**
```bash
# SSH to server and run
htop              # CPU and memory usage
df -h            # Disk usage
iostat           # I/O statistics
netstat -tulpn   # Network connections
```

---

## Maintenance and Updates

### Regular Tasks

#### 1. Update Applications

```bash
# Via CapRover dashboard
Apps → [app-name] → Deployment → Deploy via Git Repository

# Or upload new TAR file
tar -czf updated-app.tar.gz .
# Upload via CapRover interface
```

#### 2. Backup Data

```bash
# Backup Grafana dashboards
# Export from Grafana UI: Settings → Export

# Backup CapRover configuration
ssh caprover-server
sudo cp -r /captain /backup/captain-$(date +%Y%m%d)

# Backup application data
docker exec cap-adversarial-detector tar -czf /backup/app-data.tar.gz /app/data
```

#### 3. Monitor Resource Usage

Set up alerts in Grafana for:
- **CPU usage** > 80%
- **Memory usage** > 90%
- **Disk usage** > 85%
- **Application errors** > 5%

#### 4. Update System

```bash
# SSH to server monthly
sudo apt update && sudo apt upgrade -y

# Update Docker
sudo apt install docker-ce docker-ce-cli containerd.io

# Restart CapRover if needed
sudo systemctl restart docker
```

### Scaling Considerations

When your application grows:

1. **Vertical Scaling:** Upgrade VM to larger machine type
2. **Horizontal Scaling:** Add more VM instances with load balancer
3. **Database Scaling:** Upgrade MongoDB Atlas tier or use sharding
4. **CDN Integration:** Add Cloudflare or Google Cloud CDN

---

## Cost Optimization

### Estimated Monthly Costs

**Google Compute Engine (e2-standard-4):**
- VM Instance: ~$120/month
- Persistent Disk (50GB SSD): ~$8/month
- External IP: ~$3/month
- **Total: ~$131/month**

**Cost Reduction Strategies:**

1. **Use Preemptible Instances** (80% cost reduction)
   ```bash
   --preemptible flag when creating VM
   Note: Instance may be terminated with 30-second notice
   ```

2. **Sustained Use Discounts** (automatic 30% reduction after 25% usage)

3. **Committed Use Discounts** (up to 57% off for 1-3 year commitments)

4. **Right-size VM** based on actual usage patterns

5. **Schedule instances** for development environments
   ```bash
   # Auto-stop instances during off-hours
   gcloud compute instances stop caprover-server --zone=us-central1-a
   ```

---

## Conclusion

You now have a production-ready deployment of the LLM Adversarial Prompt Detector running on Google Compute Engine with CapRover. This setup provides:

- **Professional deployment** with automatic SSL and monitoring
- **Scalable infrastructure** that can grow with your needs
- **Security best practices** with proper authentication and firewalls
- **Comprehensive monitoring** with Prometheus and Grafana
- **Cost-effective hosting** with Google Cloud's competitive pricing

### Next Steps

1. **Configure monitoring alerts** for proactive issue detection
2. **Set up automated backups** for data protection
3. **Implement CI/CD pipelines** for streamlined deployments
4. **Add caching layers** for improved performance
5. **Consider multi-region deployment** for high availability

### Resources

- [CapRover Documentation](https://caprover.com/docs/)
- [Google Cloud Compute Engine](https://cloud.google.com/compute)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [Project Repository](https://github.com/Andre-Williams22/LLM-Adversarial-Prompt-Detector)

---

*This guide provides a solid foundation for deploying ML applications in production. Customize the configuration based on your specific requirements and traffic patterns.*