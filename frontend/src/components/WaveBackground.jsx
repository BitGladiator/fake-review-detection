import React from 'react';

export default function WaveBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10 bg-[#F3F6FA]">
      <svg
        viewBox="0 0 1440 900"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full object-cover"
        preserveAspectRatio="none"
      >
        <defs>
          {/* Top Layer Gradient: Light Sky to Mid Blue */}
          <linearGradient id="waveTopGrad" x1="0%" y1="20%" x2="100%" y2="80%">
            <stop offset="0%" stopColor="#4A8CF7" stopOpacity="0.45" />
            <stop offset="50%" stopColor="#60A5FA" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.55" />
          </linearGradient>

          {/* Middle Wave Gradient: Bright Royal Blue */}
          <linearGradient id="waveMidGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2563EB" stopOpacity="0.85" />
            <stop offset="50%" stopColor="#1D4ED8" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#2563EB" stopOpacity="0.8" />
          </linearGradient>

          {/* Deep Ocean Base Wave Gradient: #0F3CC9 to #09248A */}
          <linearGradient id="waveBaseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0F3CC9" />
            <stop offset="55%" stopColor="#09259E" />
            <stop offset="100%" stopColor="#071B78" />
          </linearGradient>
        </defs>

        {/* 1. Upper sweeping wave: High on left (~22% y) & right (~26% y), dipping behind cards */}
        <path
          d="M 0,210 
             C 120,260 260,420 520,530 
             C 740,620 940,560 1140,420 
             C 1280,320 1380,240 1440,230 
             L 1440,900 L 0,900 Z"
          fill="url(#waveTopGrad)"
        />

        {/* 2. Middle sweeping wave: Adds depth and layer definition */}
        <path
          d="M 0,330 
             C 180,410 380,580 680,610 
             C 980,640 1180,490 1350,420 
             C 1390,400 1420,390 1440,380 
             L 1440,900 L 0,900 Z"
          fill="url(#waveMidGrad)"
        />

        {/* 3. Deep royal blue foreground wave: fills lower portion rising smoothly on right */}
        <path
          d="M 0,480 
             C 220,510 440,680 760,680 
             C 1060,680 1260,570 1440,510 
             L 1440,900 L 0,900 Z"
          fill="url(#waveBaseGrad)"
        />

        {/* 4. Single 4-Point Star Accent in the bottom-right corner */}
        <g transform="translate(1315, 755)">
          <path
            d="M 0 -25 Q 0 0 25 0 Q 0 0 0 25 Q 0 0 -25 0 Q 0 0 0 -25 Z"
            fill="#4F74D8"
            opacity="0.65"
          />
        </g>
      </svg>
    </div>
  );
}
