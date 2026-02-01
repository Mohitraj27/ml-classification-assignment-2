# Submission PDF Template

## Structure of Submission PDF

Your final submission PDF should contain the following in order:

---

## Page 1: Cover Page

```
┌─────────────────────────────────────────────────┐
│                                                 │
│                                                 │
│         MACHINE LEARNING ASSIGNMENT 2           │
│                                                 │
│              Classification Models              │
│                                                 │
│                                                 │
│           BITS Pilani - M.Tech (AIML/DSE)      │
│                                                 │
│                                                 │
│           Student Name: S M                     │
│           Student ID: [Your ID]                 │
│                                                 │
│           Course: Machine Learning              │
│           Instructor: [Instructor Name]         │
│                                                 │
│           Submission Date: 15-Feb-2026          │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Page 2: Submission Links

### 📌 Required Links

**1. GitHub Repository**
- URL: https://github.com/YOUR_USERNAME/ml-classification-assignment
- Status: Public
- Contents: Complete source code, requirements.txt, README.md

**2. Live Streamlit Application**
- URL: https://your-app-name.streamlit.app
- Status: Deployed and Running
- Features: Model training, evaluation, predictions

**3. BITS Virtual Lab Execution**
- Screenshot: [Attached on next page]
- Date of Execution: [Date]
- Status: Completed Successfully

---

## Page 3: Screenshots

### Screenshot 1: BITS Virtual Lab Execution
[Paste screenshot showing code execution in BITS Virtual Lab]

### Screenshot 2: Streamlit App Home Page
[Paste screenshot of deployed Streamlit application]

### Screenshot 3: GitHub Repository
[Paste screenshot showing repository structure and files]

---

## Pages 4+: Complete README.md Content

[Copy the entire content of your README.md file here]

Include:
- Problem Statement
- Dataset Description
- Model Performance Comparison Table
- Model Observations Table
- All sections from README

---

## Formatting Guidelines for PDF

1. **Font**: Use professional fonts (Arial, Calibri, Times New Roman)
2. **Size**: Minimum 11pt for body text, 14-16pt for headings
3. **Links**: Make sure all URLs are clickable hyperlinks
4. **Tables**: Ensure tables are clearly formatted and readable
5. **Images**: High resolution (minimum 300 DPI)
6. **Page Numbers**: Add page numbers at bottom
7. **File Size**: Keep under 10MB if possible

---

## How to Create the PDF

### Option 1: Using Microsoft Word
1. Create document with above structure
2. Format text and insert screenshots
3. Make URLs clickable (Insert → Hyperlink)
4. Save As → PDF

### Option 2: Using Google Docs
1. Create document with structure
2. Add content and screenshots
3. Format links as hyperlinks
4. File → Download → PDF

### Option 3: Using Markdown + Pandoc
```bash
# Install pandoc
# Create markdown file
pandoc submission.md -o submission.pdf --pdf-engine=xelatex
```

---

## Checklist Before Submitting PDF

- [ ] Cover page with all required information
- [ ] All three links are present and clickable
- [ ] BITS Virtual Lab screenshot is clear and visible
- [ ] Complete README.md content is included
- [ ] All tables are properly formatted
- [ ] All images are high quality
- [ ] Page numbers are added
- [ ] File name: YourName_ML_Assignment2.pdf
- [ ] File size is reasonable (< 10MB)
- [ ] PDF opens correctly on different devices
- [ ] All hyperlinks work when clicked

---

## File Naming Convention

```
StudentName_ML_Assignment2.pdf
```

Example:
```
SM_ML_Assignment2.pdf
```

---

## Submission Process

1. **Prepare PDF** following above template
2. **Verify all links** work correctly
3. **Check file size** (compress if > 10MB)
4. **Upload to Taxila** before deadline
5. **Verify submission** was successful
6. **No resubmissions** allowed!

---

## Important Notes

⚠️ **Critical**:
- Only ONE submission will be accepted
- No resubmissions allowed
- Submit before 15-Feb-2026 23:59 PM
- Links must be clickable
- Screenshots must be clear
- README content must be complete

✅ **Good Practices**:
- Submit 1-2 hours before deadline
- Test all links before submission
- Get someone else to verify PDF
- Keep backup copy of submission
- Take screenshot of successful submission

---

## Sample Link Section (Copy This Format)

### GitHub Repository
🔗 https://github.com/skmohit05/ml-classification-assignment

**Repository Contents:**
- ✅ app.py (Streamlit application)
- ✅ requirements.txt
- ✅ README.md
- ✅ model/ directory with training scripts
- ✅ All saved model files (.pkl)

### Live Application
🔗 https://ml-classification-assignment.streamlit.app

**Application Features:**
- ✅ File upload functionality
- ✅ Model training interface
- ✅ Performance comparison
- ✅ Confusion matrices
- ✅ Prediction interface

### BITS Virtual Lab
📸 Screenshot attached on Page 3
- Execution Date: [Your Date]
- All models trained successfully
- All metrics calculated correctly

---

## Final Submission Command

```bash
# Verify everything is ready
ls -la
git status
streamlit run app.py  # Test locally

# Then submit PDF to Taxila LMS
# No email submission
# Only through Taxila portal
```

---

## Support

If you face any issues:
- Email: neha.vinayak@pilani.bits-pilani.ac.in
- Subject: "ML Assignment 2: BITS Lab issue"
- Include: Student ID and specific issue

---

**Good Luck! 🚀**

Remember: Quality over speed. Double-check everything before final submission!
