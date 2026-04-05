"use client";

/**
 * Overlay.tsx
 * ──────────────────────────────────────────────────────────────────────
 * Fixed, full-viewport narrative text layer that sits on top of the
 * 3D Canvas (z-10).  pointer-events-none so scroll/mouse still reach
 * the scene.  Each text block starts at opacity: 0 and is animated
 * by the ScrollDirector master timeline via GSAP DOM targets.
 */

import { STREAMLIT_URL } from "./ScrollDirector";

export default function Overlay() {
  return (
    <div className="fixed inset-0 z-10 pointer-events-none flex items-center justify-center">
      {/* ── Stacked narrative blocks ────────────────────────────────── */}
      <div className="relative w-full max-w-5xl px-6 flex items-center justify-center">
        {/* Block 1 — Title card */}
        <div
          id="text-1"
          className="absolute inset-0 flex flex-col items-center justify-center text-center"
          style={{ opacity: 0 }}
        >
          <h2
            className="text-7xl md:text-9xl font-bold tracking-tighter text-white"
            style={{
              filter: "drop-shadow(0 0 20px rgba(16, 185, 129, 0.35))",
            }}
          >
            Omni<span className="text-emerald-400">Quant</span>
          </h2>
          <p
            className="mt-4 text-lg md:text-xl tracking-widest uppercase text-emerald-400/60 font-light"
            style={{
              filter: "drop-shadow(0 0 10px rgba(16, 185, 129, 0.2))",
            }}
          >
            Quantitative Intelligence Engine
          </p>
        </div>

        {/* Block 2 — Thesis */}
        <div
          id="text-2"
          className="absolute inset-0 flex flex-col items-center justify-center text-center"
          style={{ opacity: 0 }}
        >
          <h2
            className="text-5xl md:text-7xl font-medium text-emerald-400 leading-tight"
            style={{
              filter: "drop-shadow(0 0 15px rgba(16, 185, 129, 0.3))",
            }}
          >
            The Future of Finance.
          </h2>
          <p
            className="mt-6 max-w-xl text-lg text-slate-400 font-light leading-relaxed"
            style={{
              filter: "drop-shadow(0 0 8px rgba(16, 185, 129, 0.15))",
            }}
          >
            Machine learning meets market microstructure.
            <br />
            Every signal, quantified.
          </p>
        </div>

        {/* Block 3 — Closer */}
        <div
          id="text-3"
          className="absolute inset-0 flex flex-col items-center justify-center text-center"
          style={{ opacity: 0 }}
        >
          <h2
            className="text-5xl md:text-7xl font-semibold text-white leading-tight"
            style={{
              filter: "drop-shadow(0 0 15px rgba(16, 185, 129, 0.3))",
            }}
          >
            Intelligence
            <br />
            <span className="text-emerald-400">Redefined.</span>
          </h2>
        </div>
      </div>

      {/* ── "Skip to App" persistent CTA ────────────────────────────── */}
      <a
        id="skip-cta"
        href={STREAMLIT_URL}
        rel="noopener noreferrer"
        className="
          pointer-events-auto
          fixed bottom-8 right-8
          px-5 py-2.5 rounded-full
          text-sm font-medium tracking-wide
          text-emerald-400/80 border border-emerald-400/20
          backdrop-blur-md bg-slate-950/40
          transition-all duration-300
          hover:text-emerald-300 hover:border-emerald-400/50
          hover:bg-slate-900/60 hover:shadow-[0_0_20px_rgba(16,185,129,0.15)]
        "
        style={{
          filter: "drop-shadow(0 0 8px rgba(16, 185, 129, 0.1))",
        }}
      >
        Skip to App →
      </a>
    </div>
  );
}
