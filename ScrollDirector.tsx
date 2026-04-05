"use client";

/**
 * ScrollDirector.tsx
 * ──────────────────────────────────────────────────────────────────────
 * Lives OUTSIDE the R3F Canvas (it's a DOM component).
 * Sets up a GSAP ScrollTrigger master timeline that writes every
 * animated value into the shared scrollProxy object AND animates
 * the narrative overlay text blocks via DOM selectors.
 *
 * Phase 5 — Cinematic Exit: when the timeline reaches 100% scroll,
 * the user is auto-redirected to the Streamlit application after a
 * brief "total black" hold for a seamless cinema → app hand-off.
 */

import { useLayoutEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { scrollProxy } from "./scrollProxy";

gsap.registerPlugin(ScrollTrigger);

// ── Redirect target (change for production) ───────────────────────────
export const STREAMLIT_URL = "http://localhost:8501";

export default function ScrollDirector() {
  const hasRedirected = useRef(false);
  const redirectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useLayoutEffect(() => {
    // Reset guard on mount (handles HMR / remounts cleanly)
    hasRedirected.current = false;

    // ── Master timeline pinned to the full scroll runway ─────────────
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: "body",
        start: "top top",
        end: "bottom bottom",
        scrub: 1.5, // cinematic weight / lag

        // ── Phase 5: Cinematic Exit ─────────────────────────────────
        onUpdate: (self) => {
          if (self.progress >= 0.995 && !hasRedirected.current) {
            hasRedirected.current = true;

            // Hold on "Total Black" for 500ms so the user registers
            // the darkness before the browser jumps to the app.
            redirectTimer.current = setTimeout(() => {
              window.location.href = STREAMLIT_URL;
            }, 500);
          }
        },
      },
    });

    // ════════════════════════════════════════════════════════════════════
    //  3D SCENE CHECKPOINTS  (scrollProxy mutations)
    // ════════════════════════════════════════════════════════════════════

    // ── CP1 — "Approaching the Core" (0% → 25%) ────────────────────
    tl.to(
      scrollProxy,
      {
        cameraZ: 4,
        lightIntensity: 40,
        duration: 0.25,
        ease: "power2.inOut",
      },
      0
    );

    // ── CP2 — "Reveal the Depth" (25% → 50%) ───────────────────────
    tl.to(
      scrollProxy,
      {
        rotationX: Math.PI / 2,
        rotationY: Math.PI / 2,
        lightIntensity: 25,
        duration: 0.25,
        ease: "power1.inOut",
        immediateRender: false,
      },
      0.25
    );

    // ── CP3 — "Data Overload" (50% → 75%) ──────────────────────────
    tl.to(
      scrollProxy,
      {
        colorLerp: 1,
        bloomIntensity: 4.0,
        lightIntensity: 60,
        duration: 0.25,
        ease: "power2.in",
        immediateRender: false,
      },
      0.5
    );

    // ── CP4 — "Blast Through" (75% → 100%) ─────────────────────────
    tl.to(
      scrollProxy,
      {
        scale: 25,
        cameraZ: -5,
        fadeOpacity: 1,
        bloomIntensity: 6.0,
        duration: 0.25,
        ease: "power3.in",
        immediateRender: false,
      },
      0.75
    );

    // ── Progress tracker ────────────────────────────────────────────
    tl.to(
      scrollProxy,
      { progress: 1, duration: 1, ease: "none" },
      0
    );

    // ════════════════════════════════════════════════════════════════════
    //  NARRATIVE OVERLAY  (DOM element animations)
    // ════════════════════════════════════════════════════════════════════

    // ── TEXT 1 — "OmniQuant" title card ─────────────────────────────
    //    Fade in (0% → 10%), hold, fade out (12% → 20%)
    tl.fromTo(
      "#text-1",
      { opacity: 0, scale: 0.85, y: 30 },
      {
        opacity: 1,
        scale: 1,
        y: 0,
        duration: 0.10,
        ease: "power2.out",
      },
      0.0 // start immediately
    );
    tl.to(
      "#text-1",
      {
        opacity: 0,
        scale: 1.05,
        y: -20,
        duration: 0.08,
        ease: "power2.in",
        immediateRender: false,
      },
      0.12 // fade out finishes at 20%
    );

    // ── TEXT 2 — "The Future of Finance." ───────────────────────────
    //    Fade in (30% → 42%), hold, fade out (48% → 55%)
    tl.fromTo(
      "#text-2",
      { opacity: 0, scale: 0.9, y: 40 },
      {
        opacity: 1,
        scale: 1,
        y: 0,
        duration: 0.12,
        ease: "power2.out",
        immediateRender: false,
      },
      0.30
    );
    tl.to(
      "#text-2",
      {
        opacity: 0,
        scale: 1.05,
        y: -20,
        duration: 0.07,
        ease: "power2.in",
        immediateRender: false,
      },
      0.48
    );

    // ── TEXT 3 — "Intelligence Redefined." ──────────────────────────
    //    Fade in (70% → 80%), hold, fade out (82% → 90%)
    tl.fromTo(
      "#text-3",
      { opacity: 0, scale: 0.9, y: 40 },
      {
        opacity: 1,
        scale: 1,
        y: 0,
        duration: 0.10,
        ease: "power2.out",
        immediateRender: false,
      },
      0.70
    );
    tl.to(
      "#text-3",
      {
        opacity: 0,
        scale: 1.1,
        y: -30,
        duration: 0.08,
        ease: "power3.in",
        immediateRender: false,
      },
      0.82
    );

    // ── Skip CTA — fade out before the portal ───────────────────────
    tl.to(
      "#skip-cta",
      {
        opacity: 0,
        duration: 0.05,
        ease: "power2.in",
        immediateRender: false,
      },
      0.88
    );

    // ── VISUAL SAFETY: ensure everything is black at 95% → 100% ─────
    //    fadeOpacity already reaches 1.0 via CP4, but these explicit
    //    tweens guarantee all narrative text is fully gone.
    tl.to(
      "#text-1, #text-2, #text-3",
      {
        opacity: 0,
        duration: 0.01,
        immediateRender: false,
      },
      0.94
    );

    // ── Cleanup ─────────────────────────────────────────────────────
    return () => {
      // Cancel any pending redirect
      if (redirectTimer.current) clearTimeout(redirectTimer.current);

      tl.kill();
      ScrollTrigger.getAll().forEach((st) => st.kill());

      // Reset proxy to initial state
      scrollProxy.cameraZ = 10;
      scrollProxy.lightIntensity = 15;
      scrollProxy.rotationX = 0;
      scrollProxy.rotationY = 0;
      scrollProxy.scale = 1;
      scrollProxy.colorLerp = 0;
      scrollProxy.bloomIntensity = 2.0;
      scrollProxy.fadeOpacity = 0;
      scrollProxy.progress = 0;
    };
  }, []);

  // No DOM output — this is a pure side-effect component
  return null;
}
