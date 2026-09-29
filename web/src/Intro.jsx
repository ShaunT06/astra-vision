import { ASTRA_LOGO } from "./assets.js";
import { useEffect, useRef, useState } from "react";

const GREEN = "0,255,102";
const BOOT = [
  "SENTINEL VISION / SECURE BOOT",
  "INITIALIZING SENSOR FUSION",
  "SCANNING HORIZON",
  "NODE 07 / SECURE CHANNEL",
];
const DURATION = 5200;

// Short synthesized radar ping; no audio assets needed.
function ping(ctx) {
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = "sine";
  o.frequency.setValueAtTime(880, ctx.currentTime);
  o.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.4);
  g.gain.setValueAtTime(0.06, ctx.currentTime);
  g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
  o.connect(g).connect(ctx.destination);
  o.start();
  o.stop(ctx.currentTime + 0.5);
}

export default function Intro({ onDone }) {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const soundRef = useRef(false);
  const leavingRef = useRef(false);
  const [sound, setSound] = useState(false);
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [leaving, setLeaving] = useState(false);

  const finish = () => {
    if (leavingRef.current) return;
    leavingRef.current = true;
    setLeaving(true);
    setTimeout(onDone, 500);
  };

  useEffect(() => {
    soundRef.current = sound;
    if (sound && !audioRef.current) audioRef.current = new (window.AudioContext || window.webkitAudioContext)();
    if (sound) audioRef.current?.resume();
  }, [sound]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const blips = Array.from({ length: 7 }, () => ({
      a: Math.random() * Math.PI * 2,
      r: 0.2 + Math.random() * 0.7,
      seen: -1,
    }));
    let raf,
      size = 0,
      lastAngle = 0;
    const start = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      size = Math.min(window.innerWidth, window.innerHeight, 720);
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      canvas.style.width = canvas.style.height = `${size}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = (now) => {
      const t = now - start;
      const angle = reduced ? Math.PI / 3 : (t / 2400) * Math.PI * 2;
      const c = size / 2,
        R = size * 0.46;
      ctx.clearRect(0, 0, size, size);
      ctx.lineWidth = 1;
      ctx.strokeStyle = `rgba(${GREEN},0.28)`;
      for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(c, c, (R * i) / 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(c - R, c);
      ctx.lineTo(c + R, c);
      ctx.moveTo(c, c - R);
      ctx.lineTo(c, c + R);
      for (let d = 0; d < 360; d += 30) {
        const a = (d * Math.PI) / 180;
        ctx.moveTo(c + Math.cos(a) * R * 0.94, c + Math.sin(a) * R * 0.94);
        ctx.lineTo(c + Math.cos(a) * R, c + Math.sin(a) * R);
      }
      ctx.stroke();

      // sweep trail
      const steps = 40;
      for (let i = 0; i < steps; i++) {
        const a0 = angle - (i / steps) * 1.1,
          a1 = angle - ((i + 1) / steps) * 1.1;
        ctx.beginPath();
        ctx.moveTo(c, c);
        ctx.arc(c, c, R, a0, a1, true);
        ctx.closePath();
        ctx.fillStyle = `rgba(${GREEN},${0.22 * (1 - i / steps)})`;
        ctx.fill();
      }
      ctx.strokeStyle = `rgba(${GREEN},0.95)`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(c, c);
      ctx.lineTo(c + Math.cos(angle) * R, c + Math.sin(angle) * R);
      ctx.stroke();

      // tracking points light up when the beam passes
      const wrapped = ((angle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
      blips.forEach((b) => {
        const crossed =
          lastAngle > wrapped ? b.a >= lastAngle || b.a <= wrapped : b.a >= lastAngle && b.a <= wrapped;
        if (crossed || reduced) {
          b.seen = now;
          if (crossed && soundRef.current && audioRef.current) ping(audioRef.current);
        }
        const fade = b.seen < 0 ? 0 : Math.max(0, 1 - (now - b.seen) / 1800);
        if (fade <= 0) return;
        const x = c + Math.cos(b.a) * R * b.r,
          y = c + Math.sin(b.a) * R * b.r;
        ctx.shadowColor = `rgb(${GREEN})`;
        ctx.shadowBlur = 14;
        ctx.fillStyle = `rgba(${GREEN},${fade})`;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = `rgba(${GREEN},${fade * 0.6})`;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 9, y - 9, 18, 18);
      });
      lastAngle = wrapped;
      if (!reduced) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    const tick = setInterval(() => {
      const p = Math.min(1, (performance.now() - start) / DURATION);
      setProgress(p);
      setStep(Math.min(BOOT.length - 1, Math.floor(p * BOOT.length)));
    }, 80);
    const done = setTimeout(finish, reduced ? 1200 : DURATION);

    return () => {
      cancelAnimationFrame(raf);
      clearInterval(tick);
      clearTimeout(done);
      window.removeEventListener("resize", resize);
      audioRef.current?.close();
      audioRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      className={`intro ${leaving ? "intro--leaving" : ""}`}
      data-testid="radar-intro"
      role="dialog"
      aria-label="Sentinel Vision secure boot"
    >
      <div className="intro__grid" aria-hidden="true" />
      <canvas ref={canvasRef} className="intro__radar" data-testid="radar-canvas" aria-hidden="true" />
      <div className="intro__top">
        <span className="mono intro__title" data-testid="intro-title">
          <img src={ASTRA_LOGO} alt="ASTRA" data-testid="intro-astra-logo" />
          {BOOT[0]}
        </span>
        <button
          className="chip-btn"
          data-testid="intro-audio-toggle"
          aria-pressed={sound}
          onClick={() => setSound((s) => !s)}
        >
          AUDIO {sound ? "ON" : "OFF"}
        </button>
      </div>
      <div className="intro__bottom">
        <ul className="intro__log mono" data-testid="intro-boot-log" aria-live="polite">
          {BOOT.slice(1, step + 1).map((m) => (
            <li key={m}>
              <i /> {m}
            </li>
          ))}
        </ul>
        <div className="intro__bar" data-testid="intro-progress">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
        <button className="chip-btn" data-testid="intro-skip-button" onClick={finish} autoFocus>
          SKIP INTRO
        </button>
      </div>
    </div>
  );
}
