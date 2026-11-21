import React, { useState, useEffect, useRef } from 'react';
import { Star, ChevronDown, Check } from 'lucide-react';
import api from '../services/api';

export default function AnalyzerView({ onPredictionSaved, prefillData }) {
  const [model, setModel] = useState('xgboost'); // 'xgboost', 'linearsvc', 'sgd'
  const [rating, setRating] = useState(4.0);
  const [reviewText, setReviewText] = useState(
    'Sample ipsum dolor sit amet, consectetur adipiscing elit, sompte lee whten they vever late on the milits and disciplt to wat create cot a precomments a left pecept, and soc was everything to touch and review.'
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [showSamplesMenu, setShowSamplesMenu] = useState(false);
  const [showRatingMenu, setShowRatingMenu] = useState(false);

  const samplesMenuRef = useRef(null);
  const ratingMenuRef = useRef(null);

  // Load samples from GET /api/samples once
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const res = await api.getSamples();
        if (res && res.samples) {
          setSamples(res.samples);
        }
      } catch (err) {
        console.warn('Could not fetch samples:', err);
      }
    };
    fetchSamples();
  }, []);

  // Handle prefill from History or Get Started
  useEffect(() => {
    if (prefillData) {
      setReviewText(prefillData.text || '');
      setRating(prefillData.rating || 4.0);
      if (prefillData.model_name) {
        const m = prefillData.model_name.toLowerCase();
        if (['xgboost', 'linearsvc', 'sgd'].includes(m)) {
          setModel(m);
        }
      }
    }
  }, [prefillData]);

  // Click outside listener for dropdowns
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (samplesMenuRef.current && !samplesMenuRef.current.contains(e.target)) {
        setShowSamplesMenu(false);
      }
      if (ratingMenuRef.current && !ratingMenuRef.current.contains(e.target)) {
        setShowRatingMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelectSample = (sample) => {
    setReviewText(sample.review_text);
    setRating(sample.rating);
    setShowSamplesMenu(false);
    setError(null);
  };

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    if (!reviewText.trim() || reviewText.trim().length < 3) {
      setError('Please enter at least 3 characters.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.predictSingle({
        review_text: reviewText.trim(),
        rating: Number(rating),
        model_name: model,
      });

      setResult(response);

      // Save to localStorage
      const historyItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        text: reviewText.trim(),
        rating: Number(rating),
        model_name: model.toUpperCase(),
        verdict: response.selected_model.prediction, // "Fake" or "Genuine"
        is_spam: response.selected_model.is_spam,
        confidence: response.selected_model.confidence,
        spam_probability: response.selected_model.spam_probability,
        latency_ms: response.latency_ms,
      };

      try {
        const stored = JSON.parse(localStorage.getItem('fair_shield_history') || '[]');
        const updated = [historyItem, ...stored].slice(0, 50);
        localStorage.setItem('fair_shield_history', JSON.stringify(updated));
        if (onPredictionSaved) onPredictionSaved(historyItem);
      } catch (storageErr) {
        console.warn('localStorage error:', storageErr);
      }
    } catch (err) {
      setError(err.message || 'Prediction failed. Verify backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Values for Right Card: use active prediction result if available, else exact reference defaults
  const isGenuine = result ? result.selected_model.prediction === 'Genuine' : true;
  const confidencePct = result ? Math.round(result.selected_model.confidence * 100) : 94;
  const statusLabel = isGenuine ? 'Genuine' : 'Fake / Spam';

  // SVG Circular Ring Calculation
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (confidencePct / 100) * circumference;

  // Sentiment bar percentage (default 72% matching screenshot)
  const sentimentPct = result
    ? Math.round(((result.signals.sentiment_score + 1) / 2) * 100)
    : 72;

  // Keyword tags (active signals or reference defaults)
  const keywordTags = result && result.signals?.top_risk_signals?.length > 0
    ? result.signals.top_risk_signals.slice(0, 3)
    : ['Natural Language', 'High Engagement', 'Verified Purchase'];

  return (
    <div className="max-w-4xl mx-auto px-4">
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-stretch">
        {/* ============================================================== */}
        {/* LEFT CARD: Analyze Review (~58% width / col-span-7)            */}
        {/* ============================================================== */}
        <div className="md:col-span-7 bg-white rounded-2xl shadow-[0_15px_40px_rgba(15,23,42,0.06)] p-6 border border-slate-100 flex flex-col justify-between">
          <div>
            {/* Title: Exactly 'Analyze Review' */}
            <h2 className="text-lg font-bold text-slate-900 mb-3 font-sans">
              Analyze Review
            </h2>

            {/* Model Selector & Compact Inline Controls Row */}
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-slate-500 font-medium">
                  Model selector
                </span>

                {/* Inline Compact Sample & Rating Helpers */}
                <div className="flex items-center gap-2">
                  {/* Star Rating Picker */}
                  <div className="relative" ref={ratingMenuRef}>
                    <button
                      type="button"
                      onClick={() => setShowRatingMenu(!showRatingMenu)}
                      className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-500 hover:text-amber-600 bg-slate-50 px-2 py-0.5 rounded-lg border border-slate-200 cursor-pointer"
                    >
                      <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                      <span>{rating.toFixed(1)} ★</span>
                    </button>
                    {showRatingMenu && (
                      <div className="absolute right-0 mt-1 w-28 bg-white rounded-xl shadow-lg border border-slate-200 py-1 z-50 text-xs">
                        {[5, 4, 3, 2, 1].map((r) => (
                          <button
                            key={r}
                            type="button"
                            onClick={() => {
                              setRating(r);
                              setShowRatingMenu(false);
                            }}
                            className="w-full text-left px-3 py-1 hover:bg-slate-50 flex items-center justify-between"
                          >
                            <span>{r}.0 Stars</span>
                            {rating === r && <Check className="w-3 h-3 text-brand-600" />}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Sample Dropdown */}
                  <div className="relative" ref={samplesMenuRef}>
                    <button
                      type="button"
                      onClick={() => setShowSamplesMenu(!showSamplesMenu)}
                      className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500 hover:text-slate-800 bg-slate-50 px-2 py-0.5 rounded-lg border border-slate-200 cursor-pointer"
                    >
                      <span>Sample</span>
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    {showSamplesMenu && (
                      <div className="absolute right-0 mt-1 w-72 bg-white rounded-xl shadow-xl border border-slate-200 p-1.5 z-50 text-xs">
                        <div className="px-2 py-1 text-[10px] font-bold text-slate-400 uppercase tracking-wider border-b border-slate-100">
                          Curated Reviews
                        </div>
                        <div className="max-h-48 overflow-y-auto divide-y divide-slate-50 py-1">
                          {samples.map((s) => (
                            <button
                              key={s.id}
                              type="button"
                              onClick={() => handleSelectSample(s)}
                              className="w-full text-left p-2 hover:bg-slate-50 rounded-lg transition-colors"
                            >
                              <div className="flex items-center justify-between font-semibold text-slate-800">
                                <span>{s.title}</span>
                                <span className="text-[10px] text-amber-500">{s.rating}★</span>
                              </div>
                              <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">
                                "{s.review_text}"
                              </p>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Model Pills: XGBoost, SVC, SGD */}
              <div className="flex items-center gap-2">
                {[
                  { id: 'xgboost', label: 'XGBoost' },
                  { id: 'linearsvc', label: 'SVC' },
                  { id: 'sgd', label: 'SGD' },
                ].map((m) => {
                  const isSelected = model === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setModel(m.id)}
                      className={`px-3.5 py-1 rounded-full text-xs transition-colors cursor-pointer border ${
                        isSelected
                          ? 'bg-[#EEF2F6] border-[#CBD5E1] text-slate-900 font-semibold'
                          : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50 font-normal'
                      }`}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Clean Textarea */}
            <div className="relative">
              <textarea
                id="review-input"
                value={reviewText}
                onChange={(e) => {
                  setReviewText(e.target.value);
                  if (error) setError(null);
                }}
                placeholder="Enter review text to analyze..."
                className="w-full h-32 p-3 text-xs md:text-sm text-slate-700 bg-white border border-slate-200 rounded-xl focus:border-brand-500 focus:outline-none resize-none leading-relaxed placeholder:text-slate-400"
              />
              {error && (
                <p className="text-[11px] text-rose-600 mt-1">{error}</p>
              )}
            </div>
          </div>

          {/* Full-width Royal Blue Button */}
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={loading || !reviewText.trim()}
            className="w-full py-3 px-6 rounded-xl bg-[#1652F0] hover:bg-[#1142C4] active:scale-[0.99] text-white text-sm font-medium transition-colors cursor-pointer mt-3 disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Analyze Review'}
          </button>
        </div>

        {/* ============================================================== */}
        {/* RIGHT CARD: Analysis Result (~42% width / col-span-5)          */}
        {/* ============================================================== */}
        <div className="md:col-span-5 bg-white rounded-2xl shadow-[0_15px_40px_rgba(15,23,42,0.06)] p-6 border border-slate-100 flex flex-col justify-between">
          <div>
            {/* Title: Exactly 'Analysis Result' */}
            <h2 className="text-lg font-bold text-slate-900 mb-4 font-sans">
              Analysis Result
            </h2>

            {/* Circular Progress Ring */}
            <div className="flex flex-col items-center justify-center my-2">
              <div className="relative w-28 h-28 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  {/* Track Ring */}
                  <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    className="stroke-slate-100"
                    strokeWidth="8"
                    fill="transparent"
                  />
                  {/* Progress Ring: #0D9488 emerald or #E11D48 crimson */}
                  <circle
                    cx="50"
                    cy="50"
                    r={radius}
                    stroke={isGenuine ? '#0D9488' : '#E11D48'}
                    strokeWidth="8"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    fill="transparent"
                    className="transition-all duration-700 ease-out"
                  />
                </svg>

                {/* Inner Text: 94% + Genuine */}
                <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                  <span className="text-2xl font-bold text-slate-900 leading-tight">
                    {confidencePct}%
                  </span>
                  <span className={`text-[11px] font-semibold ${isGenuine ? 'text-[#0D9488]' : 'text-[#E11D48]'}`}>
                    {statusLabel}
                  </span>
                </div>
              </div>
            </div>

            {/* Sentiment Score Section */}
            <div className="mt-4">
              <span className="text-xs text-slate-500 font-medium block mb-1.5">
                Sentiment Score
              </span>
              <div className="w-full bg-slate-200/80 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-[#4B7BEC] h-2 rounded-full transition-all duration-500"
                  style={{ width: `${Math.max(10, Math.min(100, sentimentPct))}%` }}
                />
              </div>
            </div>

            {/* Keywords tags Section */}
            <div className="mt-4">
              <span className="text-xs text-slate-500 font-medium block mb-2">
                Keywords tags
              </span>
              <div className="flex flex-wrap gap-2">
                {keywordTags.map((tag, idx) => (
                  <span
                    key={idx}
                    className="bg-slate-100 text-slate-700 text-xs font-medium px-3 py-1 rounded-full border border-slate-200/50"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
