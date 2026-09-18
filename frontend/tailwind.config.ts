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
        // "Studio" — a light, editorial palette: warm ivory ground, ink
        // text, one restrained accent. "ak" prefix keeps it out of the way
        // of Tailwind's own gray/violet/etc scales.
        ak: {
          bg: "#f7f4ef",
          panel: "#efe9e0",
          "panel-hover": "#e6ddd0",
          border: "#ddd5c8",
          line: "#26221c",
          text: "#17140f",
          muted: "#7a7266",
          subtle: "#9b9184",
          accent: "#6b3f4f",
          "accent-soft": "#8a5a6a",
          amber: "#a97a34",
          emerald: "#4f7a5c",
          red: "#a8434b",
        },
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-body)", "system-ui", "sans-serif"],
      },
      keyframes: {
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in-up": "fade-in-up 250ms ease-out",
      },
    },
  },
  plugins: [],
};
export default config;
