import type { Config } from 'tailwindcss';
export default {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        accent: '#6C5DD3'
      }
    }
  },
  plugins: []
} satisfies Config;
