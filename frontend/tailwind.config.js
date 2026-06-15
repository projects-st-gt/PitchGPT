/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        // Inter only — SF Pro is a license violation on the web.
      },
      colors: {
        // Three semantic colors only, per the locked design:
        // accent = teal-600, warn = amber, refuse = red.
        accent: {
          DEFAULT: "#0d9488", // teal-600
          fg: "#ffffff",
          subtle: "#f0fdfa", // teal-50
        },
      },
      fontSize: {
        // Three scales only: 14 / 16 / 22 / 32
        body: ["16px", { lineHeight: "1.5" }],
        small: ["14px", { lineHeight: "1.5" }],
        h2: ["22px", { lineHeight: "1.35", letterSpacing: "-0.01em" }],
        h1: ["32px", { lineHeight: "1.2", letterSpacing: "-0.02em" }],
      },
    },
  },
  plugins: [],
};
