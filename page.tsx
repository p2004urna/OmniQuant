import ClientOnly from "../components/ClientOnly";
import VisualStage from "../components/VisualStage";
import ScrollDirector from "../components/ScrollDirector";
import ScrollIndicator from "../components/ScrollIndicator";
import Overlay from "../components/Overlay";

export default function Home() {
  return (
    <main className="relative w-full h-[400vh] bg-slate-950">
      {/* SSR Shield → 3D Canvas always behind everything */}
      <ClientOnly>
        <ScrollDirector />
        <VisualStage />
      </ClientOnly>

      {/* Cinematic narrative overlay (GSAP-driven text blocks) */}
      <ClientOnly>
        <Overlay />
      </ClientOnly>

      {/* Pulsing scroll-down chevron */}
      <ClientOnly>
        <ScrollIndicator />
      </ClientOnly>
    </main>
  );
}
