# Fake Review Detection — FastAPI Backend

High-performance REST API backend for detecting fake, deceptive, and spam reviews using trained machine learning models (XGBoost, LinearSVC, and SGD) on the YelpNYC dataset.

## Features
- **Lifespan Startup Loading**: All `.joblib` models, dual TF-IDF vectorizers (20,000 sparse n-grams), and behavioral scalers load into memory once at startup (<350MB RAM).
- **Sub-15ms Single Inference**: Live multi-model comparison across XGBoost, LinearSVC, and SGD with ensemble majority voting and explainability signals.
- **Chunked Batch Processing**: Handles large `.csv` and `.xlsx` uploads in memory-safe 5,000-row chunks for MacBook Air M2 safety.
- **REST Analytics & Visualizations**: Serves metrics from `model/artifacts/metrics.json` and high-res PNG plots from `model/plots/`.
- **CORS Enabled**: Pre-configured for Vite React (`http://localhost:5173`) and Next.js / React (`http://localhost:3000`).

---

## Quickstart

### 1. Activate Environment
From the project root:
```bash
source venv/bin/activate
```

### 2. Start the Backend Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The interactive OpenAPI docs will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, loaded models, resident memory usage |
| `GET` | `/api/metrics` | Model performance benchmarks & YelpNYC dataset metadata |
| `GET` | `/api/plots` | Index of all available evaluation charts |
| `GET` | `/api/plots/{plot_name}` | Serves evaluation PNG plots (e.g. `04_confusion_matrices_grid.png`) |
| `POST` | `/api/predict` | Real-time single review prediction with explainability signals |
| `POST` | `/api/predict/batch` | Upload `.csv` or `.xlsx` spreadsheet for batch review scanning |
| `GET` | `/api/predict/batch/export/{id}` | Download batch results as CSV |
| `GET` | `/api/samples` | 6 curated test reviews (3 genuine, 3 fake) for 1-click UI testing |

---

## Example Request (Single Prediction)

`POST /api/predict`
```json
{
  "review_text": "BEST RESTAURANT EVER!!!! AMAZING FOOD!!!! MUST VISIT!!!! 5 STARS!!!! GO NOW!!!!",
  "rating": 5.0,
  "model_name": "ensemble"
}
```

### Example Response
```json
{
  "final_verdict": "Fake",
  "consensus_confidence": 0.8187,
  "selected_model": {
    "model_name": "Ensemble (Majority Vote)",
    "prediction": "Fake",
    "is_spam": true,
    "spam_probability": 0.8187,
    "confidence": 0.8187,
    "threshold": 0.5
  },
  "all_models": [
    {
      "model_name": "XGBoost",
      "prediction": "Fake",
      "is_spam": true,
      "spam_probability": 0.6803,
      "confidence": 0.6803,
      "threshold": 0.64
    },
    {
      "model_name": "LinearSVC",
      "prediction": "Fake",
      "is_spam": true,
      "spam_probability": 0.6932,
      "confidence": 0.6932,
      "threshold": 0.63
    },
    {
      "model_name": "SGD",
      "prediction": "Fake",
      "is_spam": true,
      "spam_probability": 0.8825,
      "confidence": 0.8825,
      "threshold": 0.73
    }
  ],
  "signals": {
    "word_count": 10,
    "char_length": 79,
    "exclamation_count": 16,
    "uppercase_ratio": 1.0,
    "sentiment_score": 0.94,
    "sentiment_rating_mismatch": 0.12,
    "top_risk_signals": [
      "High exclamation count (16) indicates promotional hyperbole.",
      "High uppercase letter ratio (100.0%) suggests forced urgency or spam phrasing.",
      "Repeated generic superlative praise combined with extreme 5-star rating."
    ]
  },
  "latency_ms": 11.45
}
```
