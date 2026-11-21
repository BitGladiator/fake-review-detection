import React from 'react';

export default function HeroHeader() {
  return (
    <div className="text-center max-w-3xl mx-auto px-4 mt-6 mb-6">
      <h1 className="text-3xl md:text-[42px] font-bold tracking-tight text-slate-900 leading-tight font-sans">
        Fake Review Detection
      </h1>
      <p className="text-slate-600 text-sm md:text-base mt-2 font-normal">
        Supervised NLP authenticity verification powered by Yelp review data
      </p>
    </div>
  );
}
