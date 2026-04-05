"use client";

import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { scrollProxy } from "./scrollProxy";

const POINT_COUNT = 600;
const SPHERE_RADIUS = 4.5;
const LINE_DISTANCE_THRESHOLD = 1.8;
const EMERALD = new THREE.Color("#10b981");
const WHITE = new THREE.Color("#ffffff");
const _lerpColor = new THREE.Color();

export default function NeuralPlexus() {
  const groupRef = useRef<THREE.Group>(null!);
  const pointsRef = useRef<THREE.Points>(null!);
  const linesRef = useRef<THREE.LineSegments>(null!);

  // ── Generate all geometry data ONCE ──────────────────────────────────
  const { pointsGeometry, linesGeometry, basePositions } = useMemo(() => {
    // --- Points ---
    const positions = new Float32Array(POINT_COUNT * 3);
    const sizes = new Float32Array(POINT_COUNT);
    const phases = new Float32Array(POINT_COUNT);

    for (let i = 0; i < POINT_COUNT; i++) {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / POINT_COUNT);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = SPHERE_RADIUS * Math.cbrt(Math.random());

      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      sizes[i] = 0.02 + Math.random() * 0.04;
      phases[i] = Math.random() * Math.PI * 2;
    }

    const ptGeo = new THREE.BufferGeometry();
    ptGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    ptGeo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    ptGeo.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));

    // --- Lines ---
    const lineVerts: number[] = [];
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();

    for (let i = 0; i < POINT_COUNT; i++) {
      v1.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      for (let j = i + 1; j < POINT_COUNT; j++) {
        v2.set(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]);
        if (v1.distanceTo(v2) < LINE_DISTANCE_THRESHOLD) {
          lineVerts.push(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
        }
      }
    }

    const lnGeo = new THREE.BufferGeometry();
    lnGeo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(lineVerts, 3)
    );

    return {
      pointsGeometry: ptGeo,
      linesGeometry: lnGeo,
      basePositions: positions.slice(),
    };
  }, []);

  // ── Per-frame: read scrollProxy + idle animation ─────────────────────
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const sp = scrollProxy;

    // ── Group transform (idle spin + scroll-driven rotation) ──────────
    if (groupRef.current) {
      // Idle spin layered on top of scroll-driven rotation
      groupRef.current.rotation.y = t * 0.08 + sp.rotationY;
      groupRef.current.rotation.x = Math.sin(t * 0.04) * 0.1 + sp.rotationX;

      // Scale from scroll
      const s = sp.scale;
      groupRef.current.scale.set(s, s, s);
    }

    // ── Camera z from scroll ─────────────────────────────────────────
    state.camera.position.z = sp.cameraZ;

    // ── Per-point pulsing ────────────────────────────────────────────
    if (pointsRef.current) {
      const posAttr = pointsRef.current.geometry.getAttribute(
        "position"
      ) as THREE.BufferAttribute;
      const phaseAttr = pointsRef.current.geometry.getAttribute(
        "aPhase"
      ) as THREE.BufferAttribute;
      const arr = posAttr.array as Float32Array;

      for (let i = 0; i < POINT_COUNT; i++) {
        const phase = phaseAttr.getX(i);
        const pulse = Math.sin(t * 0.6 + phase) * 0.06;

        arr[i * 3] = basePositions[i * 3] + pulse;
        arr[i * 3 + 1] =
          basePositions[i * 3 + 1] + Math.cos(t * 0.5 + phase) * 0.08;
        arr[i * 3 + 2] = basePositions[i * 3 + 2] + pulse * 0.5;
      }

      posAttr.needsUpdate = true;

      // ── Color lerp (emerald → white) from scroll ───────────────────
      const ptMat = pointsRef.current.material as THREE.PointsMaterial;
      _lerpColor.copy(EMERALD).lerp(WHITE, sp.colorLerp);
      ptMat.color.copy(_lerpColor);
    }

    // ── Line color follows point color ───────────────────────────────
    if (linesRef.current) {
      const lnMat = linesRef.current.material as THREE.LineBasicMaterial;
      _lerpColor.copy(EMERALD).lerp(WHITE, sp.colorLerp);
      lnMat.color.copy(_lerpColor);
    }
  });

  return (
    <group ref={groupRef}>
      {/* Glowing dots */}
      <points ref={pointsRef} geometry={pointsGeometry}>
        <pointsMaterial
          color={EMERALD}
          size={0.06}
          sizeAttenuation
          transparent
          opacity={0.9}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Faint connecting lines */}
      <lineSegments ref={linesRef} geometry={linesGeometry}>
        <lineBasicMaterial
          color={EMERALD}
          transparent
          opacity={0.12}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </lineSegments>
    </group>
  );
}
