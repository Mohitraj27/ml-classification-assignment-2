# 🚀 QUICK START - ML Assignment 2

## ⚡ 5-Minute Setup

### 1. Extract Files
```bash
# All files are in: ml_assignment/
cd ml_assignment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```

**That's it!** App opens at `http://localhost:8501`

---

## 📦 What You Got

### Core Files (Must Have)
- ✅ `app.py` - Main Streamlit app
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `model/train_models.py` - Training script

### Helper Files (Bonus)
- 📓 `model/ML_Assignment_2.ipynb` - Jupyter notebook
- 🎲 `model/generate_sample_data.py` - Test data
- 📖 `DEPLOYMENT_GUIDE.md` - Deploy help
- 📄 `SUBMISSION_TEMPLATE.md` - PDF format

---

## 🎯 3-Step Deployment

### Step 1: Push to GitHub (2 min)
```bash
git init
git add .
git commit -m "ML Assignment 2"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy to Streamlit (3 min)
1. Go to: https://streamlit.io/cloud
2. Sign in with GitHub
3. New App → Select repo → Deploy

### Step 3: Get Your Links
- GitHub: `https://github.com/YOUR_USERNAME/YOUR_REPO`
- Streamlit: `https://YOUR-APP.streamlit.app`

---

## 📝 Submission PDF Format

```
┌──────────────────────────────┐
│ Page 1: Cover Page           │
│ Page 2: Links (clickable)    │
│ Page 3: Screenshots (3x)     │
│ Page 4+: README.md content   │
└──────────────────────────────┘
```

**Required Screenshots:**
1. BITS Virtual Lab execution
2. Streamlit app running
3. GitHub repository

---

## ✅ Pre-Submission Checklist

Quick verification:

```bash
# 1. Test locally
streamlit run app.py  # Should open without errors

# 2. Check files exist
ls app.py requirements.txt README.md  # All present?

# 3. Verify GitHub
# - Repository is PUBLIC
# - All files pushed
# - Link works in incognito window

# 4. Verify Streamlit
# - App loads without errors
# - Can upload CSV
# - Models train successfully
# - Link works in incognito window

# 5. Verify PDF
# - All links are clickable
# - Screenshots are clear
# - README content included
```

---

## 🆘 Quick Troubleshooting

### App won't run locally?
```bash
pip install --upgrade streamlit scikit-learn xgboost
```

### Deployment fails?
- Check `requirements.txt` has exact package names
- Verify all imports match requirements

### GitHub push fails?
```bash
git remote -v  # Check remote URL
# Should show your GitHub repo
```

---

## 📊 What the App Does

1. **Home** - Project overview
2. **Training** - Upload CSV → Train 6 models
3. **Evaluation** - Compare performance
4. **Predictions** - Use models

---

## 🎓 Models Included

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost

**All with 6 metrics**: Accuracy, AUC, Precision, Recall, F1, MCC

---

## 📊 Dataset Used

**Wine Quality (UCI)**
- 1,599 samples
- 11 features
- Binary classification

---

## ⏰ Time Allocation

- Setup & Test: 15 minutes
- GitHub Deploy: 5 minutes
- Streamlit Deploy: 10 minutes
- BITS Lab Screenshot: 10 minutes
- PDF Creation: 20 minutes
- **Total: ~60 minutes**

---

## 🎯 Assignment Score: 15 marks

- Implementation: 10 marks
- Streamlit App: 4 marks
- BITS Lab: 1 mark

---

## 📅 Important Dates

**Deadline**: 15-Feb-2026, 23:59 PM
**No extensions**
**No resubmissions**

---

## 🔥 Pro Tips

1. **Test early** - Don't wait until deadline
2. **Deploy early** - Streamlit takes time
3. **Multiple screenshots** - Take extras, pick best
4. **Backup PDF** - Save multiple copies
5. **Submit early** - Avoid last-minute issues

---

## 📧 Need Help?

**BITS Lab Issues:**
Email: neha.vinayak@pilani.bits-pilani.ac.in
Subject: "ML Assignment 2: BITS Lab issue"

---

## 🎉 Ready to Submit?

**Final Checklist:**
- [ ] App runs locally
- [ ] GitHub repo is public and complete
- [ ] Streamlit app is live
- [ ] BITS Lab screenshot taken
- [ ] PDF has all required content
- [ ] All links are clickable
- [ ] Submitted before deadline

---

**YOU'RE ALL SET! 🚀**

Good luck with your submission!

---

## 📁 File Structure Reference

```
ml_assignment/
├── app.py                 👈 RUN THIS
├── requirements.txt       👈 INSTALL THIS
├── README.md             👈 INCLUDE IN PDF
├── model/
│   ├── train_models.py
│   ├── generate_sample_data.py
│   └── ML_Assignment_2.ipynb
├── DEPLOYMENT_GUIDE.md    👈 READ THIS
├── SUBMISSION_TEMPLATE.md 👈 FOLLOW THIS
└── COMPLETE_GUIDE.md      👈 FULL DETAILS
```

---

## 🔗 Quick Links Template for PDF

Copy this for your submission:

```
GitHub Repository:
https://github.com/YOUR_USERNAME/ml-classification-assignment

Live Streamlit App:
https://your-app-name.streamlit.app

Status: ✅ All features working
Date: [Your submission date]
```

---

**That's it! You have everything you need! 🎊**
