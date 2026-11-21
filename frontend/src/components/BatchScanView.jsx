import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileSpreadsheet, 
  Download, 
  Search, 
  ChevronLeft, 
  ChevronRight, 
  AlertCircle, 
  CheckCircle2, 
  ShieldAlert, 
  ShieldCheck,
  RefreshCw,
  FileText
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import api from '../services/api';

export default function BatchScanView() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState('ensemble');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterVerdict, setFilterVerdict] = useState('ALL'); // 'ALL', 'FAKE', 'GENUINE'
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) {
      setFile(dropped);
      setError(null);
    }
  };

  const handleUploadAndScan = async () => {
    if (!file) {
      setError('Please select a .csv or .xlsx file to scan.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await api.predictBatch(file, model);
      setResult(data);
      setCurrentPage(1);
    } catch (err) {
      setError(err.message || 'Batch scanning failed. Please check file format.');
    } finally {
      setLoading(false);
    }
  };

  // Create demo mock sample CSV for immediate user testing
  const downloadSampleTemplate = () => {
    const sampleCsvContent = `review_text,rating
"Great food, friendly staff, and the ambiance was perfect for a date night. Will definitely come back!",4.0
"BEST RESTAURANT EVER!!!! AMAZING FOOD!!!! MUST VISIT!!!! 5 STARS!!!! GO NOW!!!!",5.0
"The pasta was cooked to perfection. Service was a bit slow but worth the wait.",4.0
"Perfect perfect perfect! Everything was absolutely perfect! Best experience of my life! Perfect!",5.0
"Decent place, nothing extraordinary. The burger was good but a little overpriced for the portion size.",3.0
"I was paid to write this review. The food was okay I guess. Please give 5 stars they asked me to.",5.0
"We enjoyed our dinner here on Saturday. The seafood risotto was flavorful.",4.5
"Horrible service, rude waiter and cold soup. Will never step foot here again.",1.0
`;
    const blob = new Blob([sampleCsvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_yelp_reviews_to_scan.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Filtered rows for paginated table
  const filteredRows = (result?.rows || []).filter((r) => {
    const matchesSearch = r.review_text.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesVerdict = 
      filterVerdict === 'ALL' ||
      (filterVerdict === 'FAKE' && r.is_spam) ||
      (filterVerdict === 'GENUINE' && !r.is_spam);
    return matchesSearch && matchesVerdict;
  });

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const paginatedRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  // Recharts Chart Data
  const chartData = result ? [
    { name: 'Genuine', value: result.summary.genuine_count, color: '#10B981' },
    { name: 'Fake / Spam', value: result.summary.fake_count, color: '#E11D48' },
  ] : [];

  return (
    <div className="max-w-6xl mx-auto px-4 pb-20">
      {/* Upload & Config Container */}
      <div className="bg-white/95 backdrop-blur-md rounded-2xl shadow-[0_20px_50px_rgba(15,23,42,0.08)] border border-slate-100 p-6 md:p-8 mb-8">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-5 border-b border-slate-100 mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <span>Batch Review Scanner</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-brand-50 text-brand-700 font-semibold border border-brand-200">
                Chunked 5k Parallel
              </span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Upload customer review datasets in CSV or Excel format for bulk audit & CSV export
            </p>
          </div>

          <button
            type="button"
            onClick={downloadSampleTemplate}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-semibold border border-slate-200 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Download Sample CSV Template</span>
          </button>
        </div>

        {/* Drag & Drop Upload Area */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
            file
              ? 'border-brand-500 bg-brand-50/30'
              : 'border-slate-300 hover:border-brand-400 bg-slate-50/50 hover:bg-slate-50'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv,.xlsx,.xls"
            className="hidden"
          />

          <div className="w-14 h-14 rounded-2xl bg-white shadow-sm border border-slate-200 flex items-center justify-center mx-auto mb-3 text-brand-600">
            {file ? <FileSpreadsheet className="w-7 h-7" /> : <UploadCloud className="w-7 h-7" />}
          </div>

          {file ? (
            <div>
              <p className="text-sm font-bold text-slate-900 mb-1">
                {file.name}
              </p>
              <p className="text-xs text-slate-500">
                {(file.size / 1024).toFixed(1)} KB • Click or drop another file to replace
              </p>
            </div>
          ) : (
            <div>
              <p className="text-sm font-bold text-slate-800 mb-1">
                Click to browse or drag and drop your review dataset
              </p>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Supports .csv or .xlsx with a <code className="bg-slate-200 px-1 py-0.5 rounded text-slate-700">review_text</code> column
              </p>
            </div>
          )}
        </div>

        {/* Scan Controls Bar */}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-slate-100">
          <div className="flex items-center gap-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Model:
            </span>
            <div className="inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200">
              {[
                { id: 'ensemble', label: 'Ensemble (3-Way Vote)' },
                { id: 'xgboost', label: 'XGBoost' },
                { id: 'linearsvc', label: 'SVC' },
                { id: 'sgd', label: 'SGD' },
              ].map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setModel(m.id)}
                  className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                    model === m.id
                      ? 'bg-white text-slate-900 shadow-sm border border-slate-200'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={handleUploadAndScan}
            disabled={loading || !file}
            className="px-6 py-2.5 rounded-xl bg-[#1652F0] hover:bg-[#1040C7] active:scale-95 text-white text-sm font-bold shadow-md shadow-brand-600/20 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                <span>Scanning Dataset in Chunks...</span>
              </>
            ) : (
              <>
                <span>Run Batch Scan</span>
                <UploadCloud className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* ============================================================== */}
      {/* BATCH RESULTS DASHBOARD & TABLE                                */}
      {/* ============================================================== */}
      {result && (
        <div className="space-y-8 animate-in fade-in duration-300">
          {/* Summary KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white/95 backdrop-blur-md rounded-2xl p-5 border border-slate-100 shadow-sm">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">
                Total Reviews
              </span>
              <span className="text-3xl font-extrabold text-slate-900 font-sans">
                {result.summary.total.toLocaleString()}
              </span>
              <span className="text-[11px] text-slate-500 block mt-1">
                Latency: {result.latency_ms} ms
              </span>
            </div>

            <div className="bg-white/95 backdrop-blur-md rounded-2xl p-5 border border-emerald-100 shadow-sm">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 block mb-1 flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                Genuine Reviews
              </span>
              <span className="text-3xl font-extrabold text-emerald-700 font-sans">
                {result.summary.genuine_count.toLocaleString()}
              </span>
              <span className="text-[11px] text-emerald-600 block mt-1 font-semibold">
                {(100 - result.summary.fake_percentage).toFixed(1)}% of total
              </span>
            </div>

            <div className="bg-white/95 backdrop-blur-md rounded-2xl p-5 border border-rose-100 shadow-sm">
              <span className="text-xs font-bold uppercase tracking-wider text-rose-600 block mb-1 flex items-center gap-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                Fake / Spam
              </span>
              <span className="text-3xl font-extrabold text-rose-700 font-sans">
                {result.summary.fake_count.toLocaleString()}
              </span>
              <span className="text-[11px] text-rose-600 block mt-1 font-semibold">
                {result.summary.fake_percentage}% flagged
              </span>
            </div>

            <div className="bg-white/95 backdrop-blur-md rounded-2xl p-5 border border-slate-100 shadow-sm">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">
                Avg Confidence
              </span>
              <span className="text-3xl font-extrabold text-slate-900 font-sans">
                {(result.summary.average_confidence * 100).toFixed(1)}%
              </span>
              <span className="text-[11px] text-slate-500 block mt-1">
                Model: {result.model_used}
              </span>
            </div>
          </div>

          {/* Chart + Export Container */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center bg-white/95 backdrop-blur-md rounded-2xl p-6 md:p-8 border border-slate-100 shadow-sm">
            <div className="lg:col-span-5 h-64 flex flex-col items-center justify-center">
              <h3 className="text-sm font-bold text-slate-800 mb-2">Class Distribution Breakdown</h3>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`${value} reviews`, 'Count']} />
                  <Legend verticalAlign="bottom" height={36} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="lg:col-span-7 flex flex-col justify-center space-y-4 border-t lg:border-t-0 lg:border-l border-slate-100 lg:pl-8 pt-4 lg:pt-0">
              <div>
                <h3 className="text-base font-bold text-slate-900 mb-1">
                  Ready to Export Annotated Results
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Download the full dataset enriched with AI verdict classifications, individual spam probability scores, and flag reason signals.
                </p>
              </div>

              {result.export_id && (
                <div>
                  <a
                    href={api.getBatchExportUrl(result.export_id)}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 active:scale-95 text-white text-xs md:text-sm font-bold shadow-md shadow-emerald-600/20 transition-all"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download Annotated CSV Report</span>
                  </a>
                </div>
              )}
            </div>
          </div>

          {/* Paginated Results Table */}
          <div className="bg-white/95 backdrop-blur-md rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
            {/* Table Search & Filter Toolbar */}
            <div className="p-4 md:p-5 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3">
              <div className="relative flex-1 min-w-[220px] max-w-md">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                  }}
                  placeholder="Filter by review text keyword..."
                  className="w-full pl-9 pr-4 py-2 text-xs bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 text-slate-800"
                />
              </div>

              {/* Verdict Filter Pills */}
              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl">
                {[
                  { id: 'ALL', label: `All (${result.rows.length})` },
                  { id: 'GENUINE', label: `Genuine (${result.summary.genuine_count})` },
                  { id: 'FAKE', label: `Fake (${result.summary.fake_count})` },
                ].map((f) => (
                  <button
                    key={f.id}
                    onClick={() => {
                      setFilterVerdict(f.id);
                      setCurrentPage(1);
                    }}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                      filterVerdict === f.id
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Table Elements */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50/80 text-slate-500 uppercase font-bold text-[10px] tracking-wider border-b border-slate-100">
                  <tr>
                    <th className="py-3 px-4 w-12">#</th>
                    <th className="py-3 px-4">Review Text</th>
                    <th className="py-3 px-3 w-20 text-center">Stars</th>
                    <th className="py-3 px-4 w-28 text-center">Verdict</th>
                    <th className="py-3 px-4 w-24 text-right">Spam Prob</th>
                    <th className="py-3 px-4 w-24 text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {paginatedRows.length > 0 ? (
                    paginatedRows.map((row) => (
                      <tr key={row.row_index} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-3.5 px-4 font-mono text-slate-400 font-medium">
                          {row.row_index}
                        </td>
                        <td className="py-3.5 px-4 max-w-md">
                          <p className="text-slate-800 font-medium line-clamp-2">
                            "{row.review_text}"
                          </p>
                          {row.key_signals?.length > 0 && (
                            <div className="flex gap-1 mt-1">
                              {row.key_signals.map((sig, i) => (
                                <span key={i} className="text-[10px] text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                                  {sig}
                                </span>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="py-3.5 px-3 text-center font-semibold text-amber-500">
                          {row.rating ? `${row.rating}★` : '—'}
                        </td>
                        <td className="py-3.5 px-4 text-center">
                          <span
                            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold ${
                              row.prediction === 'Genuine'
                                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                : 'bg-rose-50 text-rose-700 border border-rose-200'
                            }`}
                          >
                            {row.prediction === 'Genuine' ? (
                              <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                            ) : (
                              <AlertCircle className="w-3 h-3 text-rose-600" />
                            )}
                            {row.prediction}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right font-mono font-semibold text-slate-700">
                          {(row.spam_probability * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-4 text-right font-mono font-semibold text-slate-700">
                          {(row.confidence * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="6" className="py-8 text-center text-slate-400">
                        No reviews found matching the current search criteria.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Toolbar */}
            <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <span>
                Showing {Math.min(filteredRows.length, (currentPage - 1) * pageSize + 1)} to{' '}
                {Math.min(filteredRows.length, currentPage * pageSize)} of {filteredRows.length} reviews
              </span>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <span className="font-semibold text-slate-800">
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
