// Hero visual: what Unmuddle does, in three looping stages.
//   1. Muddle:    30 wrong answers drift around, unsorted.
//   2. Group:     they glide into clusters, one per misconception.
//   3. Unmuddle:  learners pass the check question and turn teal.

(function () {
  const root = document.getElementById("viz");
  if (!root) return;
  const stage = root.querySelector(".viz");
  const phaseText = root.querySelector(".viz-phase");
  const stepBars = [...root.querySelectorAll(".viz-steps i")];

  // Cluster centres as fractions of the box: wide screens, then narrow screens.
  const groups = [
    { label: "ChatGPT searches the web", size: 13, wide: [0.27, 0.3], narrow: [0.3, 0.2] },
    { label: "Computers can't be biased", size: 10, wide: [0.73, 0.4], narrow: [0.68, 0.47] },
    { label: "Bigger model fixes it", size: 7, wide: [0.46, 0.72], narrow: [0.34, 0.74] },
  ];
  const total = groups.reduce((n, g) => n + g.size, 0);

  // Build dots and labels.
  const dots = [];
  groups.forEach((g, gi) => {
    for (let i = 0; i < g.size; i++) {
      const el = document.createElement("span");
      el.className = "viz-dot";
      stage.appendChild(el);
      dots.push({ el, group: gi, index: i });
    }
    const label = document.createElement("span");
    label.className = "viz-label";
    label.innerHTML = `${g.label}<b>${g.size}</b>`;
    stage.appendChild(label);
    g.labelEl = label;
  });

  let w = 0, h = 0;
  function measure() {
    const r = stage.getBoundingClientRect();
    w = r.width; h = r.height;
    groups.forEach(g => { g.at = w > 480 ? g.wide : g.narrow; });
  }

  const spacing = () => Math.max(12, Math.min(16, w / 30));
  const radiusOf = g => spacing() * 0.78 * Math.sqrt(g.size);

  function place(el, x, y) { el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`; }

  function scatter() {
    const pad = 22;
    dots.forEach(d => place(d.el, pad + Math.random() * (w - 2 * pad), pad + Math.random() * (h - 2 * pad)));
  }

  // Sunflower packing keeps each cluster round and evenly spaced.
  function clusterPos(g, i) {
    const r = spacing() * 0.78 * Math.sqrt(i + 0.5);
    const a = i * 2.39996;
    return [g.at[0] * w + r * Math.cos(a), g.at[1] * h + r * Math.sin(a)];
  }

  function cluster() {
    dots.forEach(d => place(d.el, ...clusterPos(groups[d.group], d.index)));
    groups.forEach(g => {
      // Centre the label under its cluster, but keep it inside the box.
      const half = g.labelEl.offsetWidth / 2;
      const x = Math.min(Math.max(g.at[0] * w, half + 8), w - half - 8);
      const y = g.at[1] * h + radiusOf(g) + 10;
      g.labelEl.style.transform = `translate(${(x - half).toFixed(1)}px, ${y.toFixed(1)}px)`;
    });
  }

  function setPhase(i, text) {
    stepBars.forEach((b, j) => b.classList.toggle("on", j <= i));
    phaseText.textContent = text;
  }

  const timers = [];
  const later = (ms, fn) => timers.push(setTimeout(fn, ms));

  function cycle() {
    timers.splice(0).forEach(clearTimeout);
    measure();
    dots.forEach(d => { d.el.classList.remove("fixed"); d.el.classList.add("muddled"); });
    groups.forEach(g => g.labelEl.classList.remove("show", "done"));
    setPhase(0, `${total} wrong answers, unsorted`);
    scatter();
    later(1600, scatter);

    later(3400, () => {
      setPhase(1, `Grouped into ${groups.length} misconceptions`);
      cluster();
    });
    later(4600, () => groups.forEach(g => g.labelEl.classList.add("show")));

    later(7000, () => {
      setPhase(2, "Learners pass the check question");
      // About two-thirds of each group gets unmuddled, one learner at a time.
      const toFix = dots.filter(d => d.index < Math.ceil(groups[d.group].size * 0.7));
      toFix.sort(() => Math.random() - 0.5).forEach((d, k) => later(k * 110, () => d.el.classList.add("fixed")));
      later(toFix.length * 110 + 200, () => groups.forEach(g => g.labelEl.classList.add("done")));
    });

    later(13500, cycle);
  }

  function still() {
    measure();
    cluster();
    groups.forEach(g => g.labelEl.classList.add("show"));
    dots.forEach(d => d.el.classList.add(d.index < groups[d.group].size * 0.7 ? "fixed" : "muddled"));
    setPhase(2, `${total} wrong answers in ${groups.length} misconceptions`);
  }

  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Place everything once without animating, then start.
  dots.forEach(d => { d.el.style.transition = "none"; });
  measure();
  scatter();
  requestAnimationFrame(() => requestAnimationFrame(() => {
    dots.forEach(d => { d.el.style.transition = ""; });
    reduce ? still() : cycle();
  }));

  let resizeTimer;
  addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => (reduce ? still() : cycle()), 200);
  });
  // Pause when the tab is hidden, restart when it comes back.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) timers.splice(0).forEach(clearTimeout);
    else if (!reduce) cycle();
  });
})();
