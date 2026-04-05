"use client";

import { useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import NeuralPlexus from "./NeuralPlexus";
import { scrollProxy } from "./scrollProxy";

// ── Mouse-following emerald point light ────────────────────────────────
function MouseLight() {
  const lightRef = useRef<THREE.PointLight>(null!);
  const { viewport } = useThree();

  useFrame((state) => {
    if (!lightRef.current) return;

    const targetX = (state.pointer.x * viewport.width) / 2;
    const targetY = (state.pointer.y * viewport.height) / 2;

    lightRef.current.position.x = THREE.MathUtils.lerp(
      lightRef.current.position.x,
      targetX,
      0.04
    );
    lightRef.current.position.y = THREE.MathUtils.lerp(
      lightRef.current.position.y,
      targetY,
      0.04
    );
    lightRef.current.position.z = 5;

    // ── Scroll-driven intensity ─────────────────────────────────────
    lightRef.current.intensity = scrollProxy.lightIntensity;
  });

  return (
    <pointLight
      ref={lightRef}
      color="#10b981"
      intensity={15}
      decay={2}
      distance={20}
      position={[0, 0, 5]}
    />
  );
}

// ── Animated bloom wrapper ─────────────────────────────────────────────
function AnimatedBloom() {
  const bloomRef = useRef<typeof Bloom | null>(null);

  useFrame(() => {
    // The Bloom effect exposes its "intensity" on the instance
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const effect = bloomRef.current as any;
    if (effect?.intensity !== undefined) {
      effect.intensity = scrollProxy.bloomIntensity;
    }
  });

  return (
    <Bloom
      ref={bloomRef as React.RefObject<never>}
      intensity={2.0}
      luminanceThreshold={0.2}
      luminanceSmoothing={0.9}
      mipmapBlur
    />
  );
}

// ── The full visual stage ──────────────────────────────────────────────
export default function VisualStage() {
  const fadeRef = useRef<HTMLDivElement>(null!);

  // Drive the fade-out overlay from the scroll proxy
  // (runs outside R3F via requestAnimationFrame)
  const rafIdRef = useRef<number>(0);

  // We start a RAF loop to keep the overlay in sync without React rerenders
  if (typeof window !== "undefined" && rafIdRef.current === 0) {
    const tick = () => {
      if (fadeRef.current) {
        fadeRef.current.style.opacity = String(scrollProxy.fadeOpacity);
      }
      rafIdRef.current = requestAnimationFrame(tick);
    };
    rafIdRef.current = requestAnimationFrame(tick);
  }

  return (
    <div className="fixed inset-0 z-0 bg-slate-950">
      <Canvas
        flat
        camera={{ position: [0, 0, 10], fov: 50 }}
        gl={{
          antialias: false,
          alpha: false,
          powerPreference: "high-performance",
          stencil: false,
          depth: true,
        }}
        dpr={[1, 1.5]}
        style={{ width: "100%", height: "100%" }}
      >
        {/* Baseline ambient */}
        <ambientLight intensity={0.15} color="#10b981" />

        {/* Interactive cursor light (intensity driven by scroll) */}
        <MouseLight />

        {/* The procedural data nebula (rotation, scale, color driven by scroll) */}
        <NeuralPlexus />

        {/* Scroll-driven bloom glow */}
        <EffectComposer enableNormalPass={false}>
          <AnimatedBloom />
        </EffectComposer>
      </Canvas>

      {/* Fade-to-black overlay for the final checkpoint */}
      <div
        ref={fadeRef}
        className="absolute inset-0 bg-black pointer-events-none"
        style={{ opacity: 0 }}
      />
    </div>
  );
}
