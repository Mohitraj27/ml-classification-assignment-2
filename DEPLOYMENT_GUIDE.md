# Deployment Guide for Streamlit Community Cloud

## Prerequisites
1. GitHub Account
2. Streamlit Community Cloud Account (Free)
3. Completed code pushed to GitHub

## Step-by-Step Deployment Process

### Step 1: Prepare Your GitHub Repository

1. **Create a new repository on GitHub**:
   - Go to https://github.com
   - Click "New repository"
   - Name it: `ml-classification-assignment`
   - Make it Public
   - Don't initialize with README (we already have one)

2. **Push your code to GitHub**:
```bash
cd ml_assignment

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: ML Assignment 2 complete implementation"

# Add remote repository (replace with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ml-classification-assignment.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud

1. **Go to Streamlit Cloud**:
   - Visit: https://streamlit.io/cloud
   - Click "Sign up" or "Sign in"

2. **Sign in with GitHub**:
   - Choose "Continue with GitHub"
   - Authorize Streamlit to access your repositories

3. **Create New App**:
   - Click "New app" button
   - You'll see a form with three fields:
     * **Repository**: Select `YOUR_USERNAME/ml-classification-assignment`
     * **Branch**: Select `main`
     * **Main file path**: Enter `app.py`

4. **Advanced Settings (Optional)**:
   - Click "Advanced settings" if you need to:
     * Set Python version (default: 3.9 is fine)
     * Add secrets (not needed for this project)
     * Change app URL

5. **Deploy**:
   - Click "Deploy!" button
   - Wait 2-5 minutes for deployment
   - Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

### Step 3: Verify Deployment

1. **Check App Status**:
   - Green indicator = App is running
   - Red indicator = Check logs for errors

2. **Common Issues and Solutions**:

   **Issue**: `ModuleNotFoundError`
   - **Solution**: Check `requirements.txt` has all dependencies
   - Make sure versions are compatible

   **Issue**: `Memory Error`
   - **Solution**: Streamlit free tier has limited RAM
   - Reduce model size or use lighter models
   - Use `@st.cache_data` decorator

   **Issue**: `File not found`
   - **Solution**: Check file paths are relative
   - Ensure all files are committed to GitHub

3. **Test Your App**:
   - Click the URL to open your app
   - Test file upload
   - Train models
   - Check all features work

### Step 4: Get Your Links

After successful deployment, you'll have:

1. **GitHub Repository Link**:
   - `https://github.com/YOUR_USERNAME/ml-classification-assignment`

2. **Live Streamlit App Link**:
   - `https://YOUR-APP-NAME.streamlit.app`

3. **Example**:
   - GitHub: `https://github.com/skmohit05/ml-classification-assignment`
   - App: `https://ml-classification-assignment.streamlit.app`

### Step 5: Prepare Submission

Create a PDF with:

1. **Page 1**: Cover Page
   - Assignment title
   - Your name and ID
   - Course details
   - Submission date

2. **Page 2**: Links
   - GitHub Repository Link (clickable)
   - Live Streamlit App Link (clickable)

3. **Page 3**: Screenshots
   - Screenshot of BITS Virtual Lab execution
   - Screenshot of running Streamlit app
   - Screenshot of GitHub repository

4. **Page 4+**: README Content
   - Copy entire README.md content
   - Include all tables and observations

### Step 6: Managing Your Deployed App

**Access App Dashboard**:
- Go to https://share.streamlit.io/
- See all your deployed apps
- Can view logs, restart, or delete apps

**View Logs**:
- Click on your app
- Click "Manage app"
- View logs to debug issues

**Update Your App**:
- Just push changes to GitHub:
```bash
git add .
git commit -m "Update: description of changes"
git push
```
- Streamlit auto-redeploys on push

**Reboot App** (if needed):
- In Streamlit dashboard
- Click "Reboot app"
- Wait for restart

### Troubleshooting Tips

1. **App Won't Deploy**:
   - Check requirements.txt syntax
   - Verify all imports in code
   - Check Python version compatibility

2. **App Crashes**:
   - Check logs in Streamlit dashboard
   - Add error handling in code
   - Test locally first: `streamlit run app.py`

3. **Slow Performance**:
   - Use `@st.cache_data` for data loading
   - Use `@st.cache_resource` for models
   - Reduce model complexity

4. **File Upload Not Working**:
   - Check file size limits (max 200MB on free tier)
   - Handle file upload errors gracefully
   - Provide clear error messages

### Best Practices

1. **Before Deployment**:
   - Test locally thoroughly
   - Check all requirements are in requirements.txt
   - Ensure README is complete
   - Commit all necessary files

2. **After Deployment**:
   - Test all features in deployed app
   - Check on different browsers
   - Verify links work
   - Take screenshots for submission

3. **Security**:
   - Don't commit sensitive data
   - Use secrets.toml for API keys
   - Don't hardcode passwords

### Support Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Streamlit Forum**: https://discuss.streamlit.io/
- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud

### Submission Checklist

Before submitting:

- [ ] GitHub repository is public
- [ ] All code is committed and pushed
- [ ] requirements.txt is complete
- [ ] README.md is comprehensive
- [ ] Streamlit app deploys successfully
- [ ] All features work in deployed app
- [ ] BITS Virtual Lab screenshot taken
- [ ] Submission PDF prepared with all links
- [ ] Links are clickable in PDF
- [ ] README content included in PDF
- [ ] Submitted before deadline (15-Feb-2026 23:59 PM)

---

## Quick Commands Reference

```bash
# Check git status
git status

# Add files
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push origin main

# Test locally
streamlit run app.py

# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Generate requirements
pip freeze > requirements.txt
```

---

Good luck with your deployment! 🚀
