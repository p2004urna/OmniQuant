/**
 * scrollProxy.ts
 * ──────────────────────────────────────────────────────────────────────
 * A plain mutable object that GSAP ScrollTrigger writes to (from the DOM
 * world) and React Three Fiber reads from (inside useFrame).  This bridge
 * avoids React re-renders entirely — both sides operate on raw mutation.
 */

export const scrollProxy = {
  // ── Camera ──────────────────────────────────────────────────────────
  cameraZ: 10,

  // ── Point-light ────────────────────────────────────────────────────
  lightIntensity: 15,

  // ── Plexus group rotation (additive on top of the idle spin) ──────
  rotationX: 0,
  rotationY: 0,

  // ── Plexus group scale ─────────────────────────────────────────────
  scale: 1,

  // ── Material color (0 = emerald, 1 = white) ───────────────────────
  colorLerp: 0,

  // ── Bloom ──────────────────────────────────────────────────────────
  bloomIntensity: 2.0,

  // ── Fade-out overlay (0 = transparent, 1 = fully black) ───────────
  fadeOpacity: 0,

  // ── Raw scroll progress 0-1 (for UI indicators) ───────────────────
  progress: 0,
};
