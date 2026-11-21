import React, { useEffect } from 'react';
import { X, ZoomIn, Download, ExternalLink } from 'lucide-react';

export default function ImageModal({ plot, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!plot) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div 
        className="relative max-w-5xl w-full bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh] border border-slate-200 animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-slate-50/50">
          <div>
            <h3 className="text-base font-bold text-slate-900 font-sans">
              {plot.title}
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              High-resolution 300-DPI evaluation chart ({plot.filename})
            </p>
          </div>

          <div className="flex items-center gap-2">
            <a
              href={plot.url}
              target="_blank"
              rel="noreferrer"
              className="p-2 rounded-xl text-slate-500 hover:text-slate-800 hover:bg-slate-200/60 transition-colors"
              title="Open full image in new tab"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-500 hover:text-rose-600 hover:bg-rose-50 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Image Body */}
        <div className="flex-1 overflow-auto p-4 md:p-6 bg-slate-100/50 flex items-center justify-center">
          <img
            src={plot.url}
            alt={plot.title}
            className="max-h-[70vh] w-auto object-contain rounded-xl shadow-sm border border-slate-200/80 bg-white"
          />
        </div>

        {/* Modal Footer Description */}
        <div className="px-6 py-3.5 border-t border-slate-100 bg-white text-xs text-slate-600 flex items-center justify-between">
          <span className="line-clamp-2 max-w-3xl">
            {plot.description}
          </span>
          <span className="font-mono text-[11px] text-brand-600 font-semibold bg-brand-50 px-2.5 py-1 rounded-md">
            YelpNYC Benchmark
          </span>
        </div>
      </div>
    </div>
  );
}
