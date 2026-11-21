/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#F4F7FB',
          100: '#E6EFFD',
          200: '#C7DCFB',
          300: '#93C5FD',
          400: '#60A5FA',
          500: '#3B82F6',
          600: '#1652F0', // Primary Action Blue
          700: '#1040C7',
          800: '#1D4ED8',
          900: '#0A2499',
          950: '#07165A',
        },
        genuine: {
          DEFAULT: '#059669',
          light: '#10B981',
          bg: '#ECFDF5',
          border: '#A7F3D0',
        },
        fake: {
          DEFAULT: '#E11D48',
          light: '#F43F5E',
          bg: '#FFF1F2',
          border: '#FECDD3',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'wave-drift': 'waveDrift 12s ease-in-out infinite alternate',
      },
      keyframes: {
        waveDrift: {
          '0%': { transform: 'translateX(0) translateY(0)' },
          '100%': { transform: 'translateX(-20px) translateY(8px)' },
        }
      }
    },
  },
  plugins: [],
}
