import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        // Dark, performance-instrument palette. "ak" prefix keeps it out of
        // the way of Tailwind's own gray/violet/etc scales.
        ak: {
          bg: "#09070f",
          panel: "rgba(255,255,255,0.045)",
          "panel-hover": "rgba(255,255,255,0.07)",
          border: "rgba(255,255,255,0.09)",
          text: "#f4f4f6",
          muted: "#a3a1b0",
          subtle: "#6f6d80",
          violet: "#a855f7",
          cyan: "#22d3ee",
          pink: "#f472b6",
          amber: "#fbbf24",
          emerald: "#34d399",
          red: "#fb7185",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        "glow-pulse": {
          "0%": { boxShadow: "0 0 0 0 rgba(168,85,247,0.55)" },
          "70%": { boxShadow: "0 0 0 22px rgba(168,85,247,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(168,85,247,0)" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        blob: {
          "0%, 100%": { transform: "translate(0, 0) scale(1)" },
          "33%": { transform: "translate(4%, -6%) scale(1.08)" },
          "66%": { transform: "translate(-3%, 4%) scale(0.95)" },
        },
        "bar-idle": {
          "0%, 100%": { transform: "scaleY(0.25)" },
          "50%": { transform: "scaleY(0.55)" },
        },
      },
      animation: {
        "glow-pulse": "glow-pulse 900ms ease-out",
        "fade-in-up": "fade-in-up 250ms ease-out",
        blob: "blob 14s ease-in-out infinite",
        "bar-idle": "bar-idle 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
