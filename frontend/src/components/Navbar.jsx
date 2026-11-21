import React from 'react';

export default function Navbar({ activeTab, setActiveTab, onGetStarted }) {
  const navItems = [
    { id: 'analyzer', label: 'Analyzer' },
    { id: 'batch', label: 'Batch Scan' },
    { id: 'benchmarks', label: 'Benchmarks' },
    { id: 'history', label: 'History' },
  ];

  return (
    <header className="pt-5 px-4">
      <nav className="max-w-5xl mx-auto bg-white rounded-2xl shadow-[0_8px_30px_rgba(15,23,42,0.06)] px-7 py-3 flex items-center justify-between">
        {/* Left: Plain Bold Uppercase Wordmark */}
        <div 
          onClick={() => setActiveTab('analyzer')}
          className="cursor-pointer select-none"
        >
          <span className="font-bold tracking-wide text-slate-900 text-sm font-sans">
            FAIR SHIELD
          </span>
        </div>

        {/* Center: Clean Text Links */}
        <div className="hidden sm:flex items-center space-x-7">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`text-sm transition-colors cursor-pointer ${
                  isActive
                    ? 'text-slate-900 font-semibold'
                    : 'text-slate-500 hover:text-slate-800 font-medium'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {/* Right: Clean Outlined Pill Button */}
        <div>
          <button
            onClick={onGetStarted}
            className="border border-slate-200 bg-white hover:bg-slate-50 active:scale-[0.98] text-slate-800 text-sm font-medium px-5 py-2 rounded-xl transition-all cursor-pointer shadow-xs"
          >
            Get Started
          </button>
        </div>
      </nav>

      {/* Mobile Nav Bar */}
      <div className="sm:hidden flex items-center justify-around bg-white mt-2 p-1.5 rounded-xl border border-slate-100 shadow-sm max-w-sm mx-auto">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`px-2.5 py-1 text-xs ${
              activeTab === item.id ? 'text-slate-900 font-bold' : 'text-slate-500'
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>
    </header>
  );
}
