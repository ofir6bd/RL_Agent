# 🎯 RL Agent UI - Quick Start Guide

A complete dynamic web-based dashboard for monitoring and evaluating your RL Agent radiation therapy treatment planning system.

## ⚡ 30-Second Start

### For Windows Users:
```bash
double-click run_ui.bat
```

### For Mac/Linux Users:
```bash
chmod +x run_ui.sh
./run_ui.sh
```

Then open `ui.html` in your browser.

---

## 📋 What You Get

This UI system includes:

### ✅ Features Implemented

1. **Dynamic Patient Selection**
   - Dropdown to browse all available patients
   - Random patient selection button
   - Real-time patient list from your `data/processed/` folder

2. **Treatment Visualization (3 PTV Buckets)**
   - PTV70, PTV63, PTV56 coverage tracking
   - Progress bars showing delivery percentage
   - Color-coded status indicators:
     - 🟢 Green: >95% coverage (excellent)
     - 🟡 Orange: 80-95% (acceptable) 
     - 🔴 Red: <80% (underdose - needs attention)

3. **OAR Monitoring (5 Organs)**
   - Real-time dose tracking for:
     - Brainstem (54 Gy limit)
     - SpinalCord (45 Gy limit)
     - Mandible (70 Gy limit)
     - LeftParotid (26 Gy limit)
     - RightParotid (26 Gy limit)
   - Violation detection with alerts
   - Dose vs. tolerance comparison

4. **Fraction-by-Fraction Progress**
   - Interactive timeline showing all fractions
   - Click any fraction to see detailed metrics
   - Real-time reward tracking per fraction

5. **Dose Maps per Beam**
   - 9 interactive beam selectors
   - Heatmap visualization showing dose distribution
   - Color-coded intensity (cool to hot colors)

6. **Agent Monitoring Dashboard**
   - Real-time reward metrics
   - OAR penalty vs. PTV reward breakdown
   - Action statistics (mean intensity, max intensity)
   - Reward trend charts

7. **Additional Useful GUI Elements**
   - Header with system status
   - Health indicator showing agent readiness
   - Treatment progress summary
   - Performance metrics cards
   - Responsive design (desktop, tablet, mobile)

---

## 🚀 Installation Steps

### Step 1: Install Dependencies (One-Time Only)

```bash
pip install -r requirements_ui.txt
```

This installs:
- Flask & CORS support
- Existing project dependencies (torch, numpy, scipy, etc.)

### Step 2: Start the Backend Server

**Option A: Using startup scripts (Recommended)**

Windows:
```bash
run_ui.bat
```

Mac/Linux:
```bash
./run_ui.sh
```

**Option B: Manual command**

```bash
python app.py --config configs/default.yaml --ckpt runs/best.pt --port 5000
```

You should see:
```
Initializing backend...
Backend ready!
 * Running on http://0.0.0.0:5000
```

### Step 3: Open the Frontend

Open `ui.html` in your web browser:
- File path: `file:///C:/Users/YourName/Desktop/RL_Agent/ui.html` (Windows)
- Or use Python's simple HTTP server:
  ```bash
  python -m http.server 8000 --directory .
  # Then visit: http://localhost:8000/ui.html
  ```

---

## 📖 Using the Dashboard

### Workflow

1. **Select a Patient**
   ```
   Patient Selection dropdown → Choose from patients
   OR
   "🎲 Random Patient" button → Get random patient
   ```

2. **Run Simulation**
   ```
   Click "▶️ Run Simulation"
   Wait for evaluation to complete
   See results in dashboard
   ```

3. **Explore Results**
   
   | Section | What to Look For |
   |---------|-----------------|
   | **PTV Coverage** | Are all PTVs getting their prescribed dose? |
   | **OAR Monitoring** | Any organs exceeding tolerance? |
   | **Fraction Timeline** | How does quality improve over fractions? |
   | **Dose Maps** | Visual distribution of dose per beam |
   | **Reward Trend** | Is the agent improving over treatment? |
   | **Performance Stats** | Is the agent using beams efficiently? |

### Key Metrics to Monitor

- **Reward Score**: Higher = better plan quality
  - Positive = Good PTV coverage
  - Negative = OAR constraint violated

- **OAR Penalty**: Should be near zero (< 0.1)
  - >0.5 = Significant violations
  - >1.0 = Severe violations

- **PTV Reward**: Should be positive and increasing
  - Indicates coverage is improving per fraction

- **Action Intensity**: Shows how hard the agent is "working"
  - Too high (>0.7) = Potential toxicity
  - Too low (<0.1) = Potential underdose

---

## 🎨 Dashboard Layout

```
┌─────────────────────────────────────────────────┐
│          HEADER: System Status & Config         │
├─────────────────────────────────────────────────┤
│                                                  │
│    CONTROL PANEL: Patient Selection & Run       │
│                                                  │
├─────────────────────────────────────────────────┤
│                                                  │
│    ┌──────────────┐  ┌──────────────────────┐   │
│    │ PTV Coverage │  │ OAR Monitoring       │   │
│    ├──────────────┤  ├──────────────────────┤   │
│    │ PTV70: 95%  │  │ Brainstem: 42/54 Gy │   │
│    │ PTV63: 88%  │  │ SpinalCord: 38/45  │   │
│    │ PTV56: 92%  │  │ [etc...]           │   │
│    └──────────────┘  └──────────────────────┘   │
│                                                  │
│    ┌──────────────┐  ┌──────────────────────┐   │
│    │Fraction      │  │ Reward Trend        │   │
│    │Metrics       │  │ (Line Chart)        │   │
│    └──────────────┘  └──────────────────────┘   │
│                                                  │
│    FRACTION TIMELINE: [Fx1] [Fx2] [Fx3] ...    │
│                                                  │
│    BEAM SELECTOR: [1] [2] [3] .... [9]         │
│    ┌─────────────────────────────────────────┐ │
│    │     DOSE HEATMAP (Selected Beam)       │ │
│    │     (Interactive Canvas)                │ │
│    └─────────────────────────────────────────┘ │
│                                                  │
│    AGENT PERFORMANCE: Stats Summary Cards       │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Files Created

| File | Purpose |
|------|---------|
| `app.py` | Flask REST API backend |
| `ui.html` | React-based interactive dashboard |
| `requirements_ui.txt` | Python dependencies |
| `UI_README.md` | Full documentation |
| `run_ui.bat` | Windows launcher script |
| `run_ui.sh` | Mac/Linux launcher script |
| `QUICK_START.md` | This file |

---

## 🐛 Common Issues & Fixes

### Issue: "Cannot connect to server"
```
Error: Failed to load data: Error: NetworkError when attempting to fetch resource

Fix 1: Check if Flask is running
→ Open http://localhost:5000/api/health in browser
→ Should see: {"status": "ok", "timestamp": "..."}

Fix 2: Change port if 5000 is in use
→ python app.py --port 5001

Fix 3: Check firewall
→ Windows: Allow Python through firewall
→ Mac: System Preferences > Security & Privacy
```

### Issue: "No patients found"
```
Error: Patient list is empty

Causes:
→ Data not preprocessed (run scripts/preprocess.py first)
→ Wrong data split folder
→ Missing dose_influence_matrix files

Fix: Check that this folder exists and has subfolders
→ data/processed/validation/pt_xxx/ (with dose_influence_matrix.npz)
```

### Issue: "CORS Error"
```
Error: Access to XMLHttpRequest blocked by CORS policy

Fix: CORS is enabled in app.py, but ensure:
→ Frontend is accessing http://localhost:5000
→ Not mixing http/https
→ Backend Flask server is running
```

### Issue: "Slow simulation"
```
Simulation takes >5 seconds per fraction

Solutions:
→ Close other programs
→ Use GPU (if available): check config.device = "cuda"
→ Reduce simulation fractions in app.py line ~150
```

---

## 🔌 API Reference (Advanced)

All endpoints return JSON. Base URL: `http://localhost:5000/api`

### Get Available Patients
```bash
GET /patients
→ {"patients": ["pt_201", "pt_202", ...], "total_count": 42}
```

### Get System Config
```bash
GET /config
→ {
    "n_fractions": 35,
    "n_beams": 9,
    "ptv_names": ["PTV70", "PTV63", "PTV56"],
    "oar_names": ["Brainstem", "SpinalCord", ...],
    "ptv_prescriptions": {"PTV70": 70.0, ...},
    "oar_tolerances": {"Brainstem": 54.0, ...}
}
```

### Get Random Patient
```bash
GET /patients/random
→ {"patient_id": "pt_245"}
```

### Run Simulation
```bash
POST /patients/pt_201/simulate
→ {
    "patient_id": "pt_201",
    "fractions": [
        {
            "fraction": 1,
            "reward": 0.847,
            "oar_penalty": 0.123,
            "ptv_reward": 0.970,
            "action_mean": 0.345,
            "action_max": 0.892
        },
        ...
    ],
    "total_fractions_simulated": 5
}
```

---

## 📊 Example Dashboard Workflow

### Scenario: Evaluating New Patient Plan

1. **Load System**
   ```
   ✓ Start server (run_ui.bat)
   ✓ Open ui.html
   ✓ See system status: "System Active"
   ```

2. **Select Patient**
   ```
   Dropdown → pt_245
   ```

3. **Run Simulation**
   ```
   Click "Run Simulation"
   Wait for 5 fractions to evaluate (~5-10 seconds)
   ```

4. **Review Results**
   ```
   Timeline shows: Fraction 1-5
   PTV Coverage:
   - PTV70: 94% ✓ (Good)
   - PTV63: 97% ✓ (Excellent)
   - PTV56: 96% ✓ (Excellent)
   
   OAR Status:
   - Brainstem: 42/54 Gy ✓ OK
   - SpinalCord: 40/45 Gy ✓ OK
   - Mandible: 65/70 Gy ✓ OK
   - Left Parotid: 22/26 Gy ✓ OK
   - Right Parotid: 24/26 Gy ✓ OK
   
   Reward: 0.847 (Good!)
   OAR Penalty: 0.023 (Minimal)
   ```

5. **Explore Beams**
   ```
   Click Beam 3 → See heatmap
   Click Beam 7 → See different angle
   Notice how dose is distributed
   ```

6. **Track Progress**
   ```
   Timeline shows reward trend
   Compare Fx1 vs Fx5
   Agent learning visible in metrics
   ```

---

## 💡 Customization Tips

### Change Color Theme
Edit `ui.html`, search for `:root {`:
```css
:root {
    --primary: #your-color;      /* Main color - change this */
    --secondary: #your-color;    /* Success color */
    --danger: #your-color;       /* Error color */
}
```

### Add New Metrics
Find the `Dashboard` component in `ui.html`, add new `<div className="card">`:
```jsx
<div className="card">
    <div className="card-title">
        <span className="card-icon">📊</span>
        Your Metric
    </div>
    {/* Your content */}
</div>
```

### Modify Simulation Depth
In `app.py`, line ~145, change `min(5, n_fractions)`:
```python
while not patient_done and fraction_idx < min(10, n_fractions):  # Show 10 fractions
```

---

## 🔐 Security Notes

⚠️ **For Development Only**

Current setup has:
- ✗ No authentication
- ✗ CORS enabled (localhost only)
- ✗ No HTTPS

**Before deploying to production:**
- Add user authentication
- Restrict CORS to specific origins
- Enable HTTPS
- Add database for result persistence
- Implement access control

---

## 📞 Need Help?

1. **Check Flask output**
   ```bash
   # Run with debug mode for more info
   python app.py --debug
   ```

2. **Check browser console**
   ```
   F12 → Console tab → Check for JavaScript errors
   ```

3. **Test API manually**
   ```bash
   curl http://localhost:5000/api/health
   ```

4. **Review full documentation**
   ```
   Read: UI_README.md
   ```

---

## 🎉 What's Next?

After getting the basic UI running:

1. **Customize for Your Workflow**
   - Adjust color scheme to match your institution
   - Add custom metrics relevant to your research
   - Modify heatmap visualization

2. **Integrate with Your Pipeline**
   - Connect to your treatment planning system
   - Export results to DICOM
   - Compare with traditional planning

3. **Advanced Features**
   - 3D dose visualization (THREE.js)
   - DVH curve plotting
   - Multi-patient comparison
   - Real-time plan optimization

4. **Deployment**
   - Set up web server (Nginx, Apache)
   - Add database backend
   - Implement user auth
   - Deploy to cloud

---

**Version**: 1.0  
**Created**: June 2026  
**Status**: ✅ Ready to Use
