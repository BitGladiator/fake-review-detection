import React, { useState, useEffect } from 'react';
import { 
  Trash2, 
  RotateCcw, 
  Clock, 
  Search, 
  CheckCircle2, 
  AlertCircle, 
  ShieldCheck, 
  ShieldAlert, 
  Download,
  Star
} from 'lucide-react';

export default function HistoryView({ onReloadReview }) {
  const [history, setHistory] = useState([]);
  const [search, setSearch] = useState('');
  const [verdictFilter, setVerdictFilter] = useState('ALL');

  const loadHistory = () => {
    try {
      const saved = JSON.parse(localStorage.getItem('fair_shield_history') || '[]');
      setHistory(saved);
    } catch (e) {
      setHistory([]);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear your local review analysis history?')) {
      localStorage.removeItem('fair_shield_history');
      setHistory([]);
    }
  };

  const handleExportHistory = () => {
    if (history.length === 0) return;
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fair_shield_history_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const filtered = history.filter((item) => {
    const textMatches = (item.text || '').toLowerCase().includes(search.toLowerCase());
    const verdictMatches = 
      verdictFilter === 'ALL' ||
      (verdictFilter === 'GENUINE' && item.verdict === 'Genuine') ||
      (verdictFilter === 'FAKE' && item.verdict === 'Fake');
    return textMatches && verdictMatches;
  });

  return (
    <div className="max-w-6xl mx-auto px-4 pb-20">
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-[0_20px_50px_rgba(15,23,42,0.08)] border border-slate-100 p-6 md:p-8">
        {/* Header toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-slate-100 mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <span>Analysis History</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 font-semibold">
                {history.length} Saved
              </span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Locally persisted audit log of all reviews evaluated in the Analyzer tab
            </p>
          </div>

          <div className="flex items-center gap-2">
            {history.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={handleExportHistory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200 transition-colors"
                >
                  <Download className="w-3.5 h-3.5 text-slate-500" />
                  <span>Export JSON</span>
                </button>
                <button
                  type="button"
                  onClick={handleClearHistory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold border border-rose-200 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5 text-rose-500" />
                  <span>Clear History</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Filter Toolbar */}
        {history.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div className="relative flex-1 min-w-[220px] max-w-md">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search history by review keyword..."
                className="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 text-slate-800"
              />
            </div>

            {/* Verdict Filter Pills */}
            <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl">
              {[
                { id: 'ALL', label: `All (${history.length})` },
                { id: 'GENUINE', label: `Genuine (${history.filter(h => h.verdict === 'Genuine').length})` },
                { id: 'FAKE', label: `Fake (${history.filter(h => h.verdict === 'Fake').length})` },
              ].map((f) => (
                <button
                  key={f.id}
                  onClick={() => setVerdictFilter(f.id)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    verdictFilter === f.id
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* History List */}
        {filtered.length > 0 ? (
          <div className="divide-y divide-slate-100">
            {filtered.map((item) => {
              const isGenuine = item.verdict === 'Genuine';
              const dateStr = new Date(item.timestamp).toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              });

              return (
                <div 
                  key={item.id} 
                  className="py-4 hover:bg-slate-50/60 p-3 rounded-xl transition-colors flex flex-col md:flex-row md:items-center justify-between gap-4"
                >
                  <div className="flex-1 min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold ${
                          isGenuine
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-rose-50 text-rose-700 border border-rose-200'
                        }`}
                      >
                        {isGenuine ? (
                          <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                        ) : (
                          <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
                        )}
                        {item.verdict}
                      </span>

                      <span className="text-xs font-semibold text-slate-800 bg-slate-100 px-2 py-0.5 rounded-md">
                        {item.model_name}
                      </span>

                      {item.rating && (
                        <span className="text-xs font-semibold text-amber-500 inline-flex items-center gap-0.5">
                          <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                          {item.rating}
                        </span>
                      )}

                      <span className="text-[11px] text-slate-400 flex items-center gap-1 ml-auto md:ml-0">
                        <Clock className="w-3 h-3" />
                        {dateStr}
                      </span>
                    </div>

                    <p className="text-xs text-slate-700 leading-relaxed line-clamp-2">
                      "{item.text}"
                    </p>
                  </div>

                  <div className="flex items-center gap-4 shrink-0">
                    <div className="text-right">
                      <span className="text-sm font-extrabold text-slate-900 block font-mono">
                        {(item.confidence * 100).toFixed(1)}%
                      </span>
                      <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                        Confidence
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={() => onReloadReview(item)}
                      className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-brand-50 hover:bg-brand-100 text-brand-700 text-xs font-bold border border-brand-200 transition-colors"
                      title="Load this review back into the Analyzer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      <span>Reload</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          /* Empty State */
          <div className="py-16 text-center text-slate-400">
            <div className="w-16 h-16 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center mx-auto mb-3 text-slate-400">
              <Clock className="w-8 h-8 stroke-[1.5]" />
            </div>
            <h3 className="text-sm font-bold text-slate-700 mb-1">
              No History Found
            </h3>
            <p className="text-xs text-slate-500 max-w-xs mx-auto">
              Reviews evaluated in the Analyzer tab will automatically appear here with their classification verdicts.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
