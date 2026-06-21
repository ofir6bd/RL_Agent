# 🎯 RL Agent UI System - Getting Started

## 👋 Welcome!

I've built you a **complete dynamic web-based dashboard** for your RL Agent radiation therapy treatment planning system. Here's how to get started in 3 minutes.

---

## ⚡ Quick Start (Choose Your Path)

### 🟢 I want to start immediately (Windows)
```bash
1. Double-click: run_ui.bat
2. Wait 10 seconds
3. Open ui.html in browser
Done! 🎉
```

### 🟢 I want to start immediately (Mac/Linux)
```bash
1. Terminal: chmod +x run_ui.sh
2. Terminal: ./run_ui.sh
3. Open ui.html in browser
Done! 🎉
```

### 🟡 I want to understand what I'm getting first
👉 Read: **PROJECT_SUMMARY.md** (5 min read)

### 🔵 I want detailed instructions
👉 Read: **QUICK_START.md** (10 min read)

### 🔴 Something isn't working
👉 Skip to: **Troubleshooting** (below)

---

## 📋 Complete Documentation Index

### Quick Reference Files

| File | Purpose | Read Time |
|------|---------|-----------|
| **START HERE →** | This file | 2 min |
| **PROJECT_SUMMARY.md** | What was built, features overview | 5 min |
| **QUICK_START.md** | Step-by-step setup & usage | 10 min |
| **UI_README.md** | Full technical documentation | 20 min |

### Code Files (for developers)

| File | Purpose | Size |
|------|---------|------|
| `app.py` | Flask backend API | 250 lines |
| `ui.html` | React dashboard frontend | 900 lines |
| `test_setup.py` | System verification | 200 lines |

### Helper Scripts

| File | Purpose | Platform |
|------|---------|----------|
| `run_ui.bat` | Launch backend (Windows) | Windows |
| `run_ui.sh` | Launch backend (Mac/Linux) | Mac/Linux |
| `requirements_ui.txt` | Python dependencies | All |

---

## 🎯 What You Get

### Dashboard Features

✅ **Patient Selection**
- Browse all available patients in dropdown
- Random patient picker
- Live patient count

✅ **3 PTV Coverage Tracking**
- PTV70, PTV63, PTV56
- Progress bars with color coding
- Prescription dose limits

✅ **5 OAR Monitoring**
- Brainstem, SpinalCord, Mandible, LeftParotid, RightParotid
- Dose vs. tolerance tracking
- Violation detection

✅ **Fraction-by-Fraction Progress**
- Interactive timeline (35 fractions)
- Per-fraction metrics
- Reward tracking

✅ **Dose Maps per Beam**
- All 9 beams displayed
- Heatmap visualization
- Color-coded intensity

✅ **Agent Performance Monitoring**
- Real-time reward metrics
- OAR/PTV penalty breakdown
- Action statistics
- Trend charts

✅ **Professional UI**
- Responsive design (mobile/tablet/desktop)
- Modern glassmorphism styling
- Real-time animations
- Accessibility features

---

## 🚀 Installation

### Option 1: Automated (Recommended)

**Windows:**
```bash
double-click: run_ui.bat
```

**Mac/Linux:**
```bash
./run_ui.sh
```

### Option 2: Manual

```bash
1. pip install -r requirements_ui.txt
2. python app.py --config configs/default.yaml --ckpt runs/best.pt
3. Open ui.html in browser
```

### Option 3: With Verification

```bash
1. python test_setup.py        # Verify everything is OK
2. python app.py --debug        # Start with debug output
3. Open http://localhost:5000/api/health  # Test API
4. Open ui.html in browser
```

---

## 📖 Usage Workflow

1. **Start System**
   - Run launcher script or Flask app
   - Open ui.html in browser

2. **Select Patient**
   - Use dropdown OR "Random" button
   - System loads patient data

3. **Run Simulation**
   - Click "Run Simulation"
   - Wait for evaluation (5-10 seconds)

4. **Explore Results**
   - View PTV coverage
   - Check OAR constraints
   - Click fractions for details
   - Inspect dose maps

5. **Analyze Metrics**
   - Review reward trends
   - Compare fractions
   - Evaluate agent performance

---

## 🎨 Dashboard Sections

### Header
- System status indicator
- Configuration summary
- Patient/beam count display

### Control Panel
- Patient selection dropdown
- Random patient button
- Run simulation button

### Main Dashboard (3-4 columns, responsive)

**Column 1:**
- PTV Coverage Status
- OAR Dose Monitoring

**Column 2:**
- Fraction Metrics
- Reward Trend Chart

**Full Width:**
- Fraction Timeline (interactive badges)
- Beam Selector (9 buttons)
- Dose Heatmap (2D visualization)
- Agent Performance Summary

---

## 🔍 File Overview

### Essential Files

**ui.html** (Frontend Dashboard)
- Modern React-based interface
- All-in-one file (no build needed)
- ~900 lines of code
- Responsive design
- Real-time visualizations

**app.py** (Backend API)
- Flask REST server
- Patient data management
- Agent simulation interface
- ~250 lines of code
- Self-contained

### Documentation

**QUICK_START.md**
- Quick reference guide
- Common issues & fixes
- API examples
- **Best for getting started**

**UI_README.md**
- Complete documentation
- Full API reference
- Architecture details
- Customization guide
- **Best for deep understanding**

**PROJECT_SUMMARY.md**
- What was built
- Feature overview
- Quick verification
- **Best for overview**

### Utilities

**test_setup.py**
- Verifies Python packages
- Checks file structure
- Validates data paths
- Diagnoses problems

**run_ui.bat / run_ui.sh**
- One-click startup
- Auto-dependency checking
- Works on Windows/Mac/Linux

---

## 🆘 Quick Troubleshooting

### Problem: "Can't connect to backend"
```
Solution 1: Check if Flask is running
  → Open http://localhost:5000/api/health in browser
  → Should show: {"status": "ok", ...}

Solution 2: Check port
  → Port 5000 might be in use
  → Try: python app.py --port 5001
```

### Problem: "No patients in dropdown"
```
Cause: Data not preprocessed

Fix:
  1. Run: python scripts/preprocess.py
  2. Or check data/processed/validation/ exists
  3. Verify it has subdirectories like "pt_201"
```

### Problem: "Simulation takes too long"
```
Solutions:
  - Close other programs
  - Use GPU if available
  - Reduce fractions simulated (in app.py)
```

### Problem: "Port already in use"
```
Solution:
  python app.py --port 5001
  (replace 5000 with different port)
```

### Problem: "CORS Error in console"
```
Solution: This is usually OK if:
  - Backend is running
  - You're accessing locally
  - Check browser console (F12) for exact error
```

**Still stuck?** → Read QUICK_START.md Troubleshooting section

---

## 📊 System Requirements

### Minimum
- Python 3.8+
- 200 MB free disk space
- Modern web browser
- 1 GB RAM

### Recommended
- Python 3.9+
- 500 MB free disk space
- Chrome/Firefox/Safari/Edge
- 2+ GB RAM
- GPU (optional, but recommended)

### Dependencies
- Flask 3.0
- PyTorch 1.9+
- NumPy
- SciPy
- YAML

(Install all with: `pip install -r requirements_ui.txt`)

---

## 🎯 Key Concepts

### PTV (Planning Target Volumes)
- Tumors or regions to treat
- Project has 3: PTV70, PTV63, PTV56
- Numbers indicate prescription dose (Gy)
- UI shows coverage percentage

### OAR (Organs At Risk)
- Healthy structures to protect
- Project has 5: Brainstem, SpinalCord, Mandible, LeftParotid, RightParotid
- Each has tolerance dose limit
- UI flags violations

### Fractions
- Single radiation therapy session
- Treatment delivered over 35 fractions
- Agent optimizes each fraction
- UI shows progression

### Beams
- Radiation directions
- Project uses 9 beams
- Each beam has 16×16 beamlets
- UI visualizes dose per beam

### Dose Map
- 2D visualization of intensity
- Color represents dose level
- Shows spatial distribution
- Interactive per-beam view

---

## 🎓 Documentation Roadmap

**5-Minute Path:**
1. Read this file (you're here)
2. Double-click launcher script
3. Open ui.html
4. Follow on-screen flow

**15-Minute Path:**
1. Read PROJECT_SUMMARY.md
2. Run test_setup.py
3. Read QUICK_START.md first 50 lines
4. Start system
5. Explore dashboard

**30-Minute Deep Dive:**
1. Read PROJECT_SUMMARY.md
2. Read QUICK_START.md completely
3. Read UI_README.md architecture section
4. Run test_setup.py with debug
5. Start system with --debug flag
6. Explore API docs in UI_README.md

**Developer Path:**
1. Read PROJECT_SUMMARY.md
2. Read UI_README.md completely
3. Review app.py (Flask backend)
4. Review ui.html (React frontend)
5. Customize for your needs

---

## 🚀 Next Steps After Setup

1. **Explore the Dashboard**
   - Try different patients
   - Click through fractions
   - Inspect dose maps
   - Review metrics

2. **Customize for Your Needs**
   - Adjust colors in ui.html
   - Add custom metrics
   - Modify data sources
   - Extend API

3. **Integrate with Workflow**
   - Export results
   - Compare plans
   - Archive evaluations
   - Share with team

4. **Deploy to Production** (if needed)
   - Add authentication
   - Enable HTTPS
   - Deploy to server
   - Set up database

---

## 📞 Where to Get Help

### Quick Reference
- **System doesn't start**: See Quick Troubleshooting above
- **Usage questions**: Read QUICK_START.md
- **Technical details**: Read UI_README.md
- **API examples**: Read UI_README.md → API section
- **Code customization**: Read UI_README.md → Development section

### Verify Installation
```bash
python test_setup.py
```

### Test API Manually
```bash
curl http://localhost:5000/api/health
curl http://localhost:5000/api/config
curl http://localhost:5000/api/patients
```

### Debug Mode
```bash
python app.py --debug
```

### Browser Console
Press F12 → Console tab → Check for JavaScript errors

---

## ✅ Pre-Flight Checklist

Before you start, make sure you have:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Project dependencies installed (`pip install -r requirements_ui.txt`)
- [ ] Processed data exists (`data/processed/validation/` with pt_xxx folders)
- [ ] Model checkpoint exists (`runs/best.pt`)
- [ ] Modern web browser available
- [ ] Port 5000 is available (or willing to use different port)

**Run this to verify:**
```bash
python test_setup.py
```

---

## 🎉 Let's Go!

You're ready to use your new dashboard!

### Start Now:

**Windows:**
```
1. Double-click: run_ui.bat
2. Wait for "Backend ready!" message
3. Click: ui.html
4. Enjoy! 🎉
```

**Mac/Linux:**
```
1. Terminal: ./run_ui.sh
2. Wait for "Backend ready!" message
3. Open: ui.html
4. Enjoy! 🎉
```

---

## 📚 Document Quick Links

| I want to... | Read this |
|--------------|-----------|
| Get running right now | QUICK_START.md (first section) |
| Understand what I got | PROJECT_SUMMARY.md |
| Full technical docs | UI_README.md |
| Troubleshoot issues | QUICK_START.md (troubleshooting) |
| Verify everything works | test_setup.py |
| Customize the UI | UI_README.md (customization) |
| Reference API docs | UI_README.md (API section) |
| Deploy to production | UI_README.md (security section) |

---

**Version**: 1.0  
**Status**: ✅ Ready to Use  
**Created**: June 2026

**Questions?** Check the Troubleshooting section above first!  
**Still stuck?** Read QUICK_START.md then UI_README.md for comprehensive help.

---

## 🎨 One Final Thing

The UI is fully customizable! Want to:
- ✏️ Change colors? Edit CSS in ui.html
- 📊 Add metrics? Add React components in ui.html
- 🔧 Extend API? Add endpoints in app.py
- 🎯 New features? Modify React state/effects

See UI_README.md → Development section for examples.

**Ready? Let's go!** 🚀
