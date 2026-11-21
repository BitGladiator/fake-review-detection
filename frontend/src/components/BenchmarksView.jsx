import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { 
  Award, 
  TrendingUp, 
  Layers, 
  Cpu, 
  Eye, 
  Database, 
  BarChart3, 
  Maximize2,
  CheckCircle2,
  Sparkles
} from 'lucide-react';
import api from '../services/api';
import ImageModal from './ImageModal';

export default function BenchmarksView() {
  const [metrics, setMetrics] = useState(null);
  const [plots, setPlots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPlot, setSelectedPlot] = useState(null);
  const [metricFilter, setMetricFilter] = useState('all'); // 'all', 'spam_focus', 'auc_acc'

  // Default fallback static metrics in case backend is offline
  const fallbackMetrics = {
    "XGBoost_Original": { accuracy: 0.894, spam_precision: 0.425, spam_recall: 0.089, spam_f1: 0.147, roc_auc: 0.793, clf_name: "XGBoost", sampler_name: "Original", is_best: false, optimal_threshold: 0.40 },
    "XGBoost_OverSampled": { accuracy: 0.774, spam_precision: 0.246, spam_recall: 0.584, spam_f1: 0.346, roc_auc: 0.791, clf_name: "XGBoost", sampler_name: "OverSampled", is_best: true, optimal_threshold: 0.64 },
    "XGBoost_UnderSampled": { accuracy: 0.776, spam_precision: 0.245, spam_recall: 0.569, spam_f1: 0.343, roc_auc: 0.786, clf_name: "XGBoost", sampler_name: "UnderSampled", is_best: false, optimal_threshold: 0.68 },
    "LinearSVC_Original": { accuracy: 0.897, spam_precision: 0.451, spam_recall: 0.032, spam_f1: 0.060, roc_auc: 0.761, clf_name: "LinearSVC", sampler_name: "Original", is_best: false, optimal_threshold: 0.40 },
    "LinearSVC_OverSampled": { accuracy: 0.780, spam_precision: 0.233, spam_recall: 0.499, spam_f1: 0.318, roc_auc: 0.752, clf_name: "LinearSVC", sampler_name: "OverSampled", is_best: true, optimal_threshold: 0.63 },
    "LinearSVC_UnderSampled": { accuracy: 0.773, spam_precision: 0.231, spam_recall: 0.520, spam_f1: 0.320, roc_auc: 0.750, clf_name: "LinearSVC", sampler_name: "UnderSampled", is_best: false, optimal_threshold: 0.66 },
    "SGD_Original": { accuracy: 0.893, spam_precision: 0.395, spam_recall: 0.071, spam_f1: 0.120, roc_auc: 0.772, clf_name: "SGD", sampler_name: "Original", is_best: false, optimal_threshold: 0.40 },
    "SGD_OverSampled": { accuracy: 0.750, spam_precision: 0.233, spam_recall: 0.625, spam_f1: 0.339, roc_auc: 0.776, clf_name: "SGD", sampler_name: "OverSampled", is_best: true, optimal_threshold: 0.73 },
    "SGD_UnderSampled": { accuracy: 0.756, spam_precision: 0.231, spam_recall: 0.591, spam_f1: 0.332, roc_auc: 0.768, clf_name: "SGD", sampler_name: "UnderSampled", is_best: false, optimal_threshold: 0.73 },
  };

  const defaultPlotsList = [
    {
      filename: "01_class_distribution.png",
      title: "Dataset Class Distribution",
      description: "Visualizes the extreme 90:10 real-world class imbalance of YelpNYC (322k Genuine vs. 37k Spam).",
      url: api.getPlotUrl("01_class_distribution.png")
    },
    {
      filename: "02_rating_and_length_eda.png",
      title: "Rating & Review Length EDA",
      description: "Exploratory distribution showing spam polarity bi-modality (extreme 1 & 5 stars) and length differences.",
      url: api.getPlotUrl("02_rating_and_length_eda.png")
    },
    {
      filename: "03_model_comparison_metrics.png",
      title: "Model Comparison Benchmarks",
      description: "Side-by-side comparison of Accuracy, ROC-AUC, Spam Recall, and Spam F1 across all 9 model permutations.",
      url: api.getPlotUrl("03_model_comparison_metrics.png")
    },
    {
      filename: "04_confusion_matrices_grid.png",
      title: "3×3 Confusion Matrices Grid",
      description: "Heatmap grid detailing True Negatives, False Positives, False Negatives, and True Positives across all samplers.",
      url: api.getPlotUrl("04_confusion_matrices_grid.png")
    },
    {
      filename: "05_roc_and_pr_curves.png",
      title: "ROC & Precision-Recall Curves",
      description: "Overlaid ROC and PR curves showing area-under-curve discrimination on the 71,811-row test split.",
      url: api.getPlotUrl("05_roc_and_pr_curves.png")
    },
    {
      filename: "06_top_spam_features.png",
      title: "Top Spam vs. Genuine Feature Importances",
      description: "Feature weights detailing behavioral signals, exclamation markers, and lexical tokens.",
      url: api.getPlotUrl("06_top_spam_features.png")
    },
  ];

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const metricsRes = await api.getMetrics();
        if (metricsRes && metricsRes.metrics) {
          setMetrics(metricsRes.metrics);
        } else {
          setMetrics(fallbackMetrics);
        }
      } catch (err) {
        console.warn('Using fallback benchmark metrics:', err);
        setMetrics(fallbackMetrics);
      }

      try {
        const plotsRes = await api.getPlots();
        if (plotsRes && plotsRes.plots) {
          setPlots(plotsRes.plots);
        } else {
          setPlots(defaultPlotsList);
        }
      } catch (err) {
        setPlots(defaultPlotsList);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Format data for Recharts Grouped Bar Chart
  const activeMetrics = metrics || fallbackMetrics;
  const chartData = Object.keys(activeMetrics)
    .filter((k) => k !== 'best_thresholds')
    .map((k) => {
      const row = activeMetrics[k];
      const shortName = k
        .replace('_OverSampled', ' (OS)')
        .replace('_UnderSampled', ' (US)')
        .replace('_Original', ' (Orig)');
      return {
        name: shortName,
        fullName: k,
        model: row.clf_name,
        sampler: row.sampler_name,
        accuracy: +(row.accuracy * 100).toFixed(1),
        recall: +(row.spam_recall * 100).toFixed(1),
        precision: +(row.spam_precision * 100).toFixed(1),
        f1: +(row.spam_f1 * 100).toFixed(1),
        roc_auc: +(row.roc_auc * 100).toFixed(1),
        is_best: row.is_best,
        threshold: row.optimal_threshold,
      };
    });

  return (
    <div className="max-w-6xl mx-auto px-4 pb-20 space-y-10">
      {/* ============================================================== */}
      {/* 1. Header Banner & Viva Analytics Cards                        */}
      {/* ============================================================== */}
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-[0_20px_50px_rgba(15,23,42,0.08)] border border-slate-100 p-6 md:p-8">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-slate-100 mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <span>Model Benchmarks & Viva Analytics</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-semibold border border-emerald-200">
                Professor Viva Ready
              </span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Empirical evaluation across 9 model-sampler permutations evaluated on 71,811 test reviews (YelpNYC)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Focus:</span>
            <div className="inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200">
              {[
                { id: 'all', label: 'All 5 Metrics' },
                { id: 'spam_focus', label: 'Spam Recall & F1' },
                { id: 'auc_acc', label: 'Accuracy & ROC-AUC' },
              ].map((f) => (
                <button
                  key={f.id}
                  onClick={() => setMetricFilter(f.id)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    metricFilter === f.id
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 3 Best Model Highlight Badges */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="p-4 rounded-xl bg-gradient-to-br from-blue-50 to-indigo-50/40 border border-blue-200/80 shadow-xs">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-blue-900">XGBoost (OverSampled)</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-600 text-white">
                Champion Model
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-blue-900 font-sans">0.791</span>
              <span className="text-xs font-semibold text-blue-700">ROC-AUC</span>
            </div>
            <div className="mt-2 text-[11px] text-blue-800/80 flex justify-between font-mono">
              <span>F1: 0.346</span>
              <span>Recall: 58.4%</span>
              <span>Thresh: 0.64</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-50 to-teal-50/40 border border-emerald-200/80 shadow-xs">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-emerald-900">LinearSVC (OverSampled)</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-600 text-white">
                Best Margin
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-emerald-900 font-sans">0.752</span>
              <span className="text-xs font-semibold text-emerald-700">ROC-AUC</span>
            </div>
            <div className="mt-2 text-[11px] text-emerald-800/80 flex justify-between font-mono">
              <span>F1: 0.318</span>
              <span>Recall: 49.9%</span>
              <span>Thresh: 0.63</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-purple-50 to-fuchsia-50/40 border border-purple-200/80 shadow-xs">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-purple-900">SGD Classifier (OverSampled)</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-600 text-white">
                Highest Recall
              </span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-purple-900 font-sans">62.5%</span>
              <span className="text-xs font-semibold text-purple-700">Spam Recall</span>
            </div>
            <div className="mt-2 text-[11px] text-purple-800/80 flex justify-between font-mono">
              <span>ROC-AUC: 0.776</span>
              <span>F1: 0.339</span>
              <span>Thresh: 0.73</span>
            </div>
          </div>
        </div>

        {/* ============================================================== */}
        {/* Interactive Recharts Grouped Bar Chart                         */}
        {/* ============================================================== */}
        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 10, right: 10, left: -10, bottom: 25 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#475569', fontSize: 11, fontWeight: 600 }}
                interval={0}
                angle={-15}
                textAnchor="end"
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#64748B', fontSize: 11 }}
                unit="%"
              />
              <Tooltip
                formatter={(val, name) => [`${val}%`, name]}
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  boxShadow: '0 10px 25px rgba(0,0,0,0.08)',
                  fontSize: '12px',
                }}
              />
              <Legend verticalAlign="top" height={36} />

              {(metricFilter === 'all' || metricFilter === 'auc_acc') && (
                <Bar dataKey="accuracy" name="Accuracy" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              )}
              {(metricFilter === 'all' || metricFilter === 'auc_acc') && (
                <Bar dataKey="roc_auc" name="ROC-AUC" fill="#10B981" radius={[4, 4, 0, 0]} />
              )}
              {(metricFilter === 'all' || metricFilter === 'spam_focus') && (
                <Bar dataKey="recall" name="Spam Recall" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              )}
              {(metricFilter === 'all' || metricFilter === 'spam_focus') && (
                <Bar dataKey="f1" name="Spam F1" fill="#E11D48" radius={[4, 4, 0, 0]} />
              )}
              {metricFilter === 'all' && (
                <Bar dataKey="precision" name="Spam Precision" fill="#F59E0B" radius={[4, 4, 0, 0]} />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 2. Full 9-Model Comparison Metrics Table                       */}
      {/* ============================================================== */}
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-[0_20px_50px_rgba(15,23,42,0.08)] border border-slate-100 p-6 md:p-8">
        <h3 className="text-lg font-bold text-slate-900 mb-1 flex items-center gap-2">
          <span>Complete 9-Model Test Evaluation Matrix</span>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium">
            71,811 Test Rows
          </span>
        </h3>
        <p className="text-xs text-slate-500 mb-5">
          Validation metrics measured on the held-out 20% stratified YelpNYC test split under optimal F1 threshold calibration.
        </p>

        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 font-bold uppercase text-[10px] tracking-wider border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Classifier Architecture</th>
                <th className="py-3 px-3">Resampling Technique</th>
                <th className="py-3 px-3 text-center">Threshold</th>
                <th className="py-3 px-3 text-right">Accuracy</th>
                <th className="py-3 px-3 text-right">Spam Recall</th>
                <th className="py-3 px-3 text-right">Spam Precision</th>
                <th className="py-3 px-3 text-right">Spam F1</th>
                <th className="py-3 px-3 text-right">ROC-AUC</th>
                <th className="py-3 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {chartData.map((row) => (
                <tr
                  key={row.fullName}
                  className={`transition-colors ${
                    row.is_best ? 'bg-blue-50/40 font-semibold' : 'hover:bg-slate-50/60'
                  }`}
                >
                  <td className="py-3 px-4 font-bold text-slate-900 flex items-center gap-1.5">
                    {row.model}
                    {row.is_best && <Award className="w-3.5 h-3.5 text-brand-600 shrink-0" />}
                  </td>
                  <td className="py-3 px-3 text-slate-600">
                    {row.sampler}
                  </td>
                  <td className="py-3 px-3 text-center font-mono text-slate-500">
                    {row.threshold.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-medium">
                    {row.accuracy.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-purple-700">
                    {row.recall.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono">
                    {row.precision.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-rose-600">
                    {(row.f1 / 100).toFixed(3)}
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-emerald-600">
                    {(row.roc_auc / 100).toFixed(3)}
                  </td>
                  <td className="py-3 px-3 text-center">
                    {row.is_best ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-brand-100 text-brand-800">
                        <CheckCircle2 className="w-3 h-3 text-brand-600" />
                        Best {row.model}
                      </span>
                    ) : (
                      <span className="text-[10px] text-slate-400">Baseline</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 3. High-Resolution Evaluation Plot Gallery                      */}
      {/* ============================================================== */}
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-[0_20px_50px_rgba(15,23,42,0.08)] border border-slate-100 p-6 md:p-8">
        <div className="flex items-center justify-between pb-5 border-b border-slate-100 mb-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <span>Publication & Viva Evaluation Plots</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium">
                6 Visualizations
              </span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any chart to open the high-resolution 300-DPI modal inspection viewer
            </p>
          </div>
        </div>

        {/* 6 Grid Plot Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {plots.map((plot) => (
            <div
              key={plot.filename}
              onClick={() => setSelectedPlot(plot)}
              className="group bg-slate-50 hover:bg-white rounded-2xl border border-slate-200/80 p-4 transition-all duration-200 hover:shadow-lg hover:border-brand-300 cursor-pointer flex flex-col justify-between"
            >
              <div>
                <div className="relative aspect-video rounded-xl overflow-hidden bg-white border border-slate-200/80 mb-3 flex items-center justify-center">
                  <img
                    src={plot.url}
                    alt={plot.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 bg-slate-950/0 group-hover:bg-slate-950/20 transition-colors flex items-center justify-center">
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 text-slate-900 text-xs font-bold px-3 py-1.5 rounded-xl shadow-md flex items-center gap-1.5">
                      <Maximize2 className="w-3.5 h-3.5" />
                      Zoom Chart
                    </span>
                  </div>
                </div>

                <h4 className="text-sm font-bold text-slate-900 group-hover:text-brand-600 transition-colors mb-1 font-sans">
                  {plot.title}
                </h4>
                <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                  {plot.description}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                <span>{plot.filename}</span>
                <span className="text-brand-600 font-semibold group-hover:underline flex items-center gap-0.5">
                  Inspect &rarr;
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox Zoom Modal */}
      {selectedPlot && (
        <ImageModal
          plot={selectedPlot}
          onClose={() => setSelectedPlot(null)}
        />
      )}
    </div>
  );
}
