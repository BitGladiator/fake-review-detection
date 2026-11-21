import React, { useState } from 'react';
import WaveBackground from './components/WaveBackground';
import Navbar from './components/Navbar';
import HeroHeader from './components/HeroHeader';
import AnalyzerView from './components/AnalyzerView';
import BatchScanView from './components/BatchScanView';
import BenchmarksView from './components/BenchmarksView';
import HistoryView from './components/HistoryView';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [prefillData, setPrefillData] = useState(null);

  const handleGetStarted = () => {
    setActiveTab('analyzer');
    const el = document.getElementById('review-input');
    if (el) {
      el.focus();
    }
  };

  const handleReloadReview = (item) => {
    setPrefillData(item);
    setActiveTab('analyzer');
    setTimeout(() => {
      const el = document.getElementById('review-input');
      if (el) {
        el.focus();
      }
    }, 50);
  };

  return (
    <div className={`relative min-h-screen flex flex-col ${activeTab === 'analyzer' ? 'md:h-screen md:overflow-hidden justify-start' : 'justify-between'}`}>
      {/* Sweeping Layered Blue Wave Background Canvas with Bottom-Right Star */}
      <WaveBackground />

      {/* Main Content Layer */}
      <div className="relative z-10 flex-1 flex flex-col">
        {/* Floating Top Navigation Pill Bar */}
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onGetStarted={handleGetStarted}
        />

        {/* Hero Title & Subtitle */}
        <HeroHeader />

        {/* Main View Display */}
        <main className="flex-1 pb-4">
          {activeTab === 'analyzer' && (
            <AnalyzerView prefillData={prefillData} />
          )}

          {activeTab === 'batch' && (
            <BatchScanView />
          )}

          {activeTab === 'benchmarks' && (
            <BenchmarksView />
          )}

          {activeTab === 'history' && (
            <HistoryView onReloadReview={handleReloadReview} />
          )}
        </main>
      </div>
    </div>
  );
}
