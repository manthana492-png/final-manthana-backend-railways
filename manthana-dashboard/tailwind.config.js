/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: { 950: "#0a0c0f", 900: "#12151a", 800: "#1c2129", 700: "#2a3140" },
        accent: { DEFAULT: "#0d9488", muted: "#0f766e", glow: "#2dd4bf" },
        risk: { low: "#22c55e", mid: "#eab308", high: "#f97316", crit: "#ef4444" },
      },
      boxShadow: {
        panel: "0 4px 24px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};
