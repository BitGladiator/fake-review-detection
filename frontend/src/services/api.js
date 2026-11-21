/**
 * api.js — API Client for Fake Review Detection FastAPI Backend
 * Target backend: http://127.0.0.1:8000
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

class ApiService {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  async _fetch(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    try {
      const response = await fetch(url, options);
      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson && errorJson.detail) {
            errorDetail = typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
          }
        } catch {
          // ignore json parse errors on non-200
        }
        throw new Error(errorDetail);
      }
      return await response.json();
    } catch (err) {
      console.error(`API Error on ${endpoint}:`, err);
      throw err;
    }
  }

  /**
   * Health status of backend
   */
  async checkHealth() {
    return this._fetch('/api/health');
  }

  /**
   * Single review live prediction
   */
  async predictSingle({ review_text, rating = 4.0, model_name = 'xgboost', user_id = null, product_id = null }) {
    return this._fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        review_text,
        rating: Number(rating),
        model_name: model_name.toLowerCase(),
        user_id,
        product_id,
      }),
    });
  }

  /**
   * Upload CSV or Excel file for batch scanning
   */
  async predictBatch(file, modelName = 'ensemble') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', modelName.toLowerCase());

    const url = `${this.baseUrl}/api/predict/batch`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = `Batch scan failed (${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson && errorJson.detail) errorDetail = errorJson.detail;
      } catch {}
      throw new Error(errorDetail);
    }
    return await response.json();
  }

  /**
   * Get direct download URL for batch export CSV
   */
  getBatchExportUrl(exportId) {
    return `${this.baseUrl}/api/predict/batch/export/${exportId}`;
  }

  /**
   * Get model evaluation metrics and YelpNYC dataset summary
   */
  async getMetrics() {
    return this._fetch('/api/metrics');
  }

  /**
   * List all available PNG evaluation charts
   */
  async getPlots() {
    return this._fetch('/api/plots');
  }

  /**
   * Get image URL for a given plot filename
   */
  getPlotUrl(plotName) {
    const safeName = plotName.endsWith('.png') ? plotName : `${plotName}.png`;
    return `${this.baseUrl}/api/plots/${safeName}`;
  }

  /**
   * Fetch 6 curated sample reviews for 1-click testing
   */
  async getSamples() {
    return this._fetch('/api/samples');
  }
}

export const api = new ApiService();
export default api;
