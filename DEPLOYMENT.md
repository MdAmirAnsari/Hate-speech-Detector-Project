# 🚀 Deployment Guide

This guide explains how to deploy the Hate Speech Detection System to various platforms.

## 📋 Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## 🌐 Streamlit Cloud Deployment

### Step 1: Push to GitHub

1. **Create a new repository** on GitHub
2. **Clone or upload** this project to your repository
3. **Ensure all files are committed** and pushed to the `main` branch

### Step 2: Deploy to Streamlit Cloud

1. **Visit [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository** and branch (`main`)
5. **Set the main file path** to `streamlit_app.py`
6. **Click "Deploy"**

### Step 3: Configure (Optional)

- **Custom domain**: Add your own domain in the app settings
- **Secrets**: Add any sensitive configuration in the secrets section
- **Resource limits**: Monitor usage and upgrade if needed

## 🔧 Manual Deployment Options

### Option 1: Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `runtime.txt`:
   ```
   python-3.9.18
   ```

3. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 2: Railway

1. Connect your GitHub repository to Railway
2. Set the start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy automatically

### Option 3: Google Cloud Platform

1. Create a `app.yaml`:
   ```yaml
   runtime: python39
   
   handlers:
   - url: /.*
     script: auto
   
   env_variables:
     PORT: 8080
   ```

2. Deploy using `gcloud app deploy`

## 🔍 Troubleshooting

### Common Issues

1. **"No module named" errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **NLTK data not found**
   - The app automatically downloads NLTK data
   - If issues persist, add to the startup script

3. **Model loading errors**
   - Verify all `.pkl` files are in the repository
   - Check file paths are correct

4. **Memory issues**
   - Models are large (~33MB total)
   - Consider using Streamlit Cloud's resource limits

### Performance Tips

1. **Use caching**: The app uses `@st.cache_resource` for model loading
2. **Optimize models**: Consider model compression if needed
3. **Monitor usage**: Check app performance in Streamlit Cloud dashboard

## 📝 Environment Variables

If you need to set environment variables:

```bash
# For development
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# For production
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 🎯 Direct Deployment Links

### Streamlit Cloud
- **Quick Deploy**: [Deploy to Streamlit Cloud](https://share.streamlit.io)
- **Documentation**: [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)

### Alternative Platforms
- **Railway**: [Deploy to Railway](https://railway.app)
- **Heroku**: [Deploy to Heroku](https://heroku.com)
- **Google Cloud**: [Deploy to GCP](https://cloud.google.com/appengine)

## 🔗 Live Demo

Once deployed, your app will be available at:
- **Streamlit Cloud**: `https://your-username-your-repo-name-streamlit-app.streamlit.app`
- **Custom domain**: Configure in your platform settings

## 🆘 Support

If you encounter issues:
1. Check the [Streamlit Community Forum](https://discuss.streamlit.io)
2. Review the [GitHub Issues](https://github.com/streamlit/streamlit/issues)
3. Check the deployment platform's documentation

---

**Happy Deploying!** 🚀
