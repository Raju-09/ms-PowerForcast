# Deployment Guide

This guide provides step-by-step instructions for deploying the Electricity Demand Forecasting application to both Vercel and Render platforms.

## Prerequisites

- Git repository initialized (GitHub, GitLab, or Bitbucket)
- Python 3.11 compatible codebase
- All model files and data committed to the repository

## Platform Comparison

| Feature | Vercel | Render |
|---------|--------|--------|
| **Deployment Type** | Serverless | Traditional Web Server |
| **Size Limit** | 50MB per function | No strict limit |
| **TensorFlow Support** | ❌ No (too large) | ✅ Yes |
| **Free Tier** | Yes | Yes |
| **Best For** | Lightweight ML (scikit-learn only) | Full ML stack with TensorFlow |
| **Recommended** | ❌ Not recommended | ✅ **Recommended** |

## Option 1: Deploy to Render (Recommended)

Render is the **recommended platform** for this ML application because it supports TensorFlow and has no strict size limits.

### Step 1: Prepare Your Repository

Ensure all files are committed:
```bash
git add .
git commit -m "Add deployment configuration for Render"
git push origin main
```

### Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with your GitHub/GitLab account
3. Authorize Render to access your repositories

### Step 3: Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your repository
3. Configure the service:
   - **Name**: `electricity-forecasting` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or paid for better performance)

### Step 4: Set Environment Variables

In the Render dashboard, add these environment variables:

```
FLASK_ENV=production
FLASK_DEBUG=False
PYTHON_VERSION=3.11.0
```

### Step 5: Deploy

1. Click **"Create Web Service"**
2. Render will automatically build and deploy your app
3. Monitor the logs for any errors
4. Once deployed, you'll get a URL like: `https://electricity-forecasting.onrender.com`

### Step 6: Verify Deployment

Test your endpoints:
- Homepage: `https://your-app.onrender.com/`
- Predict: `https://your-app.onrender.com/predict`
- Dashboard: `https://your-app.onrender.com/dashboard`
- Forecast: `https://your-app.onrender.com/forecast`
- Anomalies: `https://your-app.onrender.com/anomalies`

### Auto-Deploy on Git Push

Render automatically redeploys when you push to your main branch. No additional configuration needed!

---

## Option 2: Deploy to Vercel (Limited Features)

> ⚠️ **WARNING**: Vercel has a 50MB size limit for serverless functions. You **MUST remove TensorFlow** to deploy on Vercel. This means **no LSTM models** will be available.

### Prerequisites for Vercel

1. **Remove TensorFlow** from your deployment:
   ```bash
   # Rename requirements.txt to requirements-render.txt
   mv requirements.txt requirements-render.txt
   
   # Use the Vercel-compatible requirements
   mv requirements-vercel.txt requirements.txt
   ```

2. **Commit changes**:
   ```bash
   git add .
   git commit -m "Switch to Vercel-compatible dependencies"
   git push origin main
   ```

### Step 1: Install Vercel CLI (Optional)

```bash
npm install -g vercel
```

### Step 2: Deploy via Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Click **"Add New Project"**
4. Import your repository
5. Configure:
   - **Framework Preset**: Other
   - **Build Command**: Leave blank
   - **Output Directory**: Leave blank
6. Click **"Deploy"**

### Step 3: Deploy via CLI (Alternative)

```bash
cd electricity-demand-forecasting
vercel
```

Follow the prompts to link your project.

### Step 4: Set Environment Variables

In Vercel dashboard:
1. Go to **Settings** → **Environment Variables**
2. Add:
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   ```

### Vercel Limitations

- ❌ No TensorFlow support
- ❌ No LSTM models
- ✅ Scikit-learn models only (Random Forest, Gradient Boosting)
- ⚠️ Cold start latency on free tier

---

## Troubleshooting

### Common Issues

#### 1. **Model files not loading**
- **Solution**: Ensure `models/` directory is committed to Git
- Check `.gitignore` doesn't exclude `.pkl` files

#### 2. **Data file not found**
- **Solution**: Ensure `data/electricity_demand.csv` is committed
- Update `.gitignore` if needed

#### 3. **Requirements installation fails**
- **Solution**: Check Python version compatibility
- Try pinning specific versions in `requirements.txt`

#### 4. **TensorFlow build timeout (Render)**
- **Solution**: This is normal. First build may take 10-15 minutes
- Subsequent builds use cache and are faster

#### 5. **Vercel: Function size too large**
- **Solution**: Remove TensorFlow dependency
- Use `requirements-vercel.txt` instead

#### 6. **Port binding error**
- **Solution**: Ensure app uses `PORT` environment variable
- Check `app.py` runs with: `port = int(os.getenv('PORT', 5000))`

### Health Check Failures

If Render health checks fail:
1. Check logs in Render dashboard
2. Verify `/` route returns HTTP 200
3. Ensure model files loaded successfully

### Build Logs

Monitor build logs to identify issues:
- **Render**: Click on the deployment → View Logs
- **Vercel**: Deployment details → Build Logs

---

## Post-Deployment

### Update `.gitignore`

Ensure your `.gitignore` doesn't exclude necessary files:
```gitignore
# Don't ignore these (needed for deployment)
!models/*.pkl
!data/*.csv
!static/
!templates/

# Do ignore these
.env
__pycache__/
*.pyc
.DS_Store
venv/
*.log
```

### Custom Domain (Optional)

Both platforms support custom domains:
- **Render**: Settings → Custom Domain
- **Vercel**: Project Settings → Domains

### Monitoring

- **Render**: Built-in logs and metrics
- **Vercel**: Analytics available on paid plans

### Performance Optimization

For better performance on free tiers:
1. **Enable caching** for static files
2. **Optimize model size** (use lighter models if possible)
3. **Add loading states** in frontend for slower responses
4. **Consider upgrading** to paid tier for production use

---

## Comparison Summary

| Aspect | Vercel | Render |
|--------|--------|--------|
| Setup Ease | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| ML Support | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Free Tier | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| For This App | ❌ Not Ideal | ✅ **Best Choice** |

## Recommendation

🎯 **Use Render** for this application. It fully supports your ML stack including TensorFlow, has no size restrictions, and provides a traditional web server environment perfect for Flask ML applications.

## Need Help?

- **Render Docs**: https://render.com/docs
- **Vercel Docs**: https://vercel.com/docs
- **Flask Deployment**: https://flask.palletsprojects.com/en/latest/deploying/

---

## Quick Reference Commands

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn (production-like)
gunicorn app:app

# Access at http://localhost:8000
```

### Git Deployment
```bash
# Stage all changes
git add .

# Commit
git commit -m "Deploy to production"

# Push (triggers auto-deploy on Render/Vercel)
git push origin main
```

---

Good luck with your deployment! 🚀
