import { useEffect, useRef, useState } from "react";
import Intro from "./Intro.jsx";
import Workbench from "./Workbench.jsx";
import { ASTRA_LOGO, HERO_POSTER, HERO_VIDEOS, IMAGES, ARCHIVE_VIDEOS } from "./assets.js";

const LINKS = [
  ["Capabilities", "capabilities", "nav-capabilities"],
  ["Application", "application", "nav-application"],
  ["About Platform", "platform", "nav-about"],
];

const CARDS = [
  { id: "aerial", n: "01 / AERIAL", title: "FIXED-WING AIRCRAFT", img: IMAGES.jet, conf: "98.7", track: "TRACK 09", sector: "SECTOR 22" },
  { id: "ground", n: "02 / GROUND", title: "ARMORED PLATFORMS", img: IMAGES.tank, conf: "96.2", track: "TRACK 14", sector: "SECTOR 07" },
  { id: "maritime", n: "03 / MARITIME", title: "NAVAL ASSETS", img: IMAGES.ship, conf: "94.8", track: "TRACK 31", sector: "SECTOR 15" },
];

const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

function Nav() {
  const [open, setOpen] = useState(false);
  const jump = (id) => {
    setOpen(false);
    go(id);
  };
  return (
    <header className="nav" data-testid="site-nav">
      <div className="nav__in wrap">
        <a className="logo" href="#top" data-testid="nav-logo" aria-label="Sentinel Vision home">
          <span className="logo__badge">SV</span>
          <span className="logo__text">SENTINEL VISION</span>
        </a>
        <img className="astra-mark" src={ASTRA_LOGO} alt="ASTRA: Armed Squad for Tactical Readiness & Awareness" data-testid="astra-logo" />
        <nav className={`nav__links ${open ? "is-open" : ""}`} aria-label="Primary" data-testid="nav-links">
          {LINKS.map(([label, id, tid]) => (
            <a key={id} href={`#${id}`} data-testid={tid} onClick={(e) => { e.preventDefault(); jump(id); }}>
              {label}
            </a>
          ))}
          <button className="btn btn--red btn--sm" data-testid="nav-open-workbench" onClick={() => jump("application")}>
            Open Workbench
          </button>
        </nav>
        <button
          className="nav__toggle"
          data-testid="nav-menu-toggle"
          aria-expanded={open}
          aria-label="Toggle menu"
          onClick={() => setOpen((o) => !o)}
        >
          <span /><span /><span />
        </button>
      </div>
    </header>
  );
}

function HeroVideo() {
  const [i, setI] = useState(0);
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  const next = () => setI((n) => (n + 1) % HERO_VIDEOS.length);
  return (
    <video
      key={HERO_VIDEOS[i]}
      className="hero__video"
      data-testid="hero-video"
      src={HERO_VIDEOS[i]}
      poster={HERO_POSTER}
      autoPlay
      muted
      playsInline
      loop={HERO_VIDEOS.length === 1}
      preload="auto"
      onEnded={next}
      onError={() => (HERO_VIDEOS.length > 1 ? next() : setFailed(true))}
    />
  );
}

function Hero() {
  return (
    <section className="hero" id="top" data-testid="hero-section">
      <div className="hero__bg" aria-hidden="true">
        <img className="hero__video" src={HERO_POSTER} alt="" />
        <HeroVideo />
        <div className="hero__shade" />
        <div className="hero__grid" />
        <div className="hero__rings" />
        <div className="hero__sweep" />
        <div className="hero__scan" />
      </div>
      <div className="hero__target mono" aria-hidden="true" data-testid="hero-target">
        <div className="hero__cross" />
        <span>SECTOR 22 / ALPHA</span>
        <span>34.0522° N · 118.2437° W</span>
        <span className="blink">TRACKING · LOCK 0.987</span>
      </div>
      <div className="wrap hero__content">
        <p className="eyebrow reveal" data-testid="hero-eyebrow"><i /> COMPUTER VISION / DEFENSE INTELLIGENCE</p>
        <h1 className="display display--xl reveal" data-testid="hero-headline">
          <span>SEE FIRST.</span>
          <br />
          <span className="red">KNOW MORE.</span>
        </h1>
        <p className="lede reveal" data-testid="hero-copy">
          A visual intelligence layer for the world's most demanding environments.
        </p>
        <button className="btn btn--red btn--lg reveal" data-testid="hero-cta-button" onClick={() => go("application")}>
          Explore the application <span aria-hidden="true">→</span>
        </button>
      </div>
      <div className="hero__foot mono" data-testid="hero-telemetry">
        <span><i className="dot" /> SCANNING 12,408 OBJECT CLASSES</span>
        <span>BUILD 2.4.1 / 2025</span>
        <span>SCROLL TO DEPLOY ↓</span>
      </div>
    </section>
  );
}

function Card({ c }) {
  const [videoOk, setVideoOk] = useState(true);
  const [hover, setHover] = useState(false);
  const ref = useRef(null);
  const src = ARCHIVE_VIDEOS[c.id];

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (hover) v.play().catch(() => {});
    else {
      v.pause();
      v.currentTime = 0;
    }
  }, [hover]);

  return (
    <article
      className="card reveal"
      data-testid={`media-card-${c.id}`}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div className="card__media">
        <img className="card__img" src={c.img} alt={c.title} loading="lazy" data-testid={`media-image-${c.id}`} />
        {src && videoOk && (
          <video
            ref={ref}
            className={`card__img card__video ${hover ? "is-on" : ""}`}
            data-testid={`media-video-${c.id}`}
            src={src}
            muted
            loop
            playsInline
            preload={hover ? "auto" : "none"}
            aria-label={`${c.title} video feed`}
            onError={() => setVideoOk(false)}
          />
        )}
        <div className="card__shade" />
        <div className="card__corners" aria-hidden="true" />
        <span className="card__n mono">{c.n}</span>
        <span className="card__live mono"><i /> LIVE FEED</span>
        <span className="card__track mono">{c.track} · {c.sector}</span>
      </div>
      <div className="card__body">
        <h3>{c.title}</h3>
        <p className="mono conf" data-testid={`confidence-${c.id}`}><b>{c.conf}%</b> CONFIDENCE</p>
      </div>
    </article>
  );
}

function Media() {
  return (
    <section id="capabilities" className="section" data-testid="media-section">
      <div className="wrap">
        <p className="eyebrow reveal"><i /> TELEMETRY ARCHIVE</p>
        <h2 className="display reveal">TELEMETRY <span className="red">ARCHIVE.</span></h2>
        <p className="lede reveal">Air, land and sea: one model family, classifying every domain.</p>
        <div className="cards">{CARDS.map((c) => <Card key={c.id} c={c} />)}</div>
        <p className="mono muted small credit">
          Imagery: US government works via Wikimedia Commons (public domain). Confidence figures are illustrative.
        </p>
      </div>
    </section>
  );
}

function About() {
  const stats = [
    ["50ms", "LATENCY TARGET"],
    ["12.4k", "OBJECT CLASSES"],
    ["99.2%", "UPTIME DESIGN"],
  ];
  return (
    <section id="platform" className="about" data-testid="about-section">
      <div className="wrap">
        <h2 className="display display--xl reveal" data-testid="about-heading">BUILT FOR THE EDGE.</h2>
        <p className="lede reveal">
          Sentinel Vision turns complex visual environments into structured operational insight: every frame
          becomes labelled objects, confidence scores and coordinates that other systems can act on.
        </p>
        <dl className="stats">
          {stats.map(([v, l]) => (
            <div key={l} className="stat reveal" data-testid={`stat-${l.split(" ")[0].toLowerCase()}`}>
              <dt className="mono">{l}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="footer mono" data-testid="site-footer">
      <div className="wrap footer__in">
        <span>SENTINEL VISION © 2025</span>
        <span>VISUAL INTELLIGENCE, DEPLOYED</span>
        <a href="/credits.csv" data-testid="credits-link">IMAGE CREDITS &amp; LICENCES</a>
        <span className="secure"><i className="dot" /> SECURE CHANNEL</span>
      </div>
    </footer>
  );
}

export default function App() {
  const [booting, setBooting] = useState(true);

  useEffect(() => {
    if (booting) return;
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))),
      { threshold: 0.12 }
    );
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [booting]);

  return (
    <>
      {booting && <Intro onDone={() => setBooting(false)} />}
      <div className={booting ? "site is-hidden" : "site"} aria-hidden={booting}>
        <Nav />
        <main>
          <Hero />
          <Media />
          <Workbench />
          <About />
        </main>
        <Footer />
      </div>
    </>
  );
}
