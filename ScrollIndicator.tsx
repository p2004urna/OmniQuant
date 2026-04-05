"use client";

import { useEffect, useRef } from "react";
import { scrollProxy } from "./scrollProxy";

/**
 * A pulsing "scroll down" chevron that fades out after 5% scroll.
 */
export default function ScrollIndicator() {
  const containerRef = useRef<HTMLDivElement>(null!);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const tick = () => {
      if (containerRef.current) {
        // Fade out over the first 5% of scroll
        const opacity = Math.max(0, 1 - scrollProxy.progress / 0.05);
        containerRef.current.style.opacity = String(opacity);
        containerRef.current.style.pointerEvents =
          opacity < 0.01 ? "none" : "auto";
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <div
      ref={containerRef}
      className="fixed bottom-10 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 pointer-events-none"
    >
      <span className="text-sm text-brand-emerald/70 tracking-widest uppercase font-light">
        Scroll
      </span>
      {/* Animated chevron */}
      <div className="scroll-chevron-container">
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="scroll-chevron"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="scroll-chevron scroll-chevron-delayed"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>
    </div>
  );
}
