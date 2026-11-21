# FAIR SHIELD — Frontend

Modern React + Vite + Tailwind CSS + Recharts user interface for **Fake Review Detection using Machine Learning**.

## Visual Design & Theme
- **Layered Wave Background**: Soft icy-white top canvas (`#F4F7FB`) transitioning into multi-layered royal blue SVG waves (`#93C5FD` → `#3B82F6` → `#1D4ED8` → `#0A2499`) with subtle mouse parallax and 4-point star accent.
- **Floating Navigation Pill**: White glassmorphic pill bar with `FAIR SHIELD` branding, view tabs (`Analyzer`, `Batch Scan`, `Benchmarks`, `History`), live FastAPI backend connection health dot, and `Get Started` quick-action button.
- **Two-Card Analyzer**:
  - Left: Interactive classifier selector (`XGBoost`, `SVC`, `SGD`), 1–5 star rating picker, quick sample loader (6 curated benchmark reviews), clean textarea, and royal blue primary action button (`#1652F0`).
  - Right: Latency badge, animated circular SVG progress ring (emerald `#0D9488` for Genuine, crimson `#E11D48` for Fake), sentiment polarity bar, explainability risk signals pill cloud, and expandable 3-model side-by-side probability comparison.
- **Batch Scanner**: Drag & drop `.csv` / `.xlsx` spreadsheet upload, chunked 5,000-row processing, Recharts breakdown donut chart, paginated results table, and annotated CSV download.
- **Viva Benchmarks**: Grouped bar chart comparing Accuracy, Spam Precision, Spam Recall, Spam F1, and ROC-AUC across all 9 runs, complete 9-model comparison table, and 6 evaluation plots gallery with click-to-zoom modal.
- **Local History**: Automatically persists past evaluations to `localStorage` with reload into analyzer functionality and JSON export.

---

## Quickstart

### 1. Ensure FastAPI Backend is Running
In one terminal:
```bash
source venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Vite Development Server
In a second terminal:
```bash
cd frontend
npm run dev
```

The frontend will start at **http://localhost:5173** and automatically connect to `http://127.0.0.1:8000`.

### 3. Production Build
```bash
cd frontend
npm run build
```
Generates production-optimized static assets in `frontend/dist/`.
