// Small helpers shared by every page.

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) {
    const message = (data && data.detail && typeof data.detail === "string")
      ? data.detail
      : "Something went wrong. Check your connection and try again.";
    throw new Error(message);
  }
  return data;
}

function esc(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function param(name) {
  return new URLSearchParams(location.search).get(name);
}

// Browser storage can be missing or blocked; never let that break the page.
const saved = {
  get(key) { try { return localStorage.getItem(key); } catch (_) { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* ignore */ } },
};

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// One dot per learner. `fixed` of them are teal, the rest amber.
function dots(total, fixed = 0, extraClass = "", attrs = "") {
  let html = `<div class="dots" role="img" aria-label="${total} learners, ${fixed} fixed" ${attrs}>`;
  for (let i = 0; i < total; i++) {
    html += `<span class="dot ${i < fixed ? "fixed" : ""} ${extraClass}"></span>`;
  }
  return html + "</div>";
}

// Update an existing dot row in place, so changes animate instead of jumping:
// new dots pop in, newly fixed dots settle into teal.
function syncDots(container, total, fixed = 0, kind = "") {
  const current = container.querySelectorAll(".dot");
  for (let i = current.length; i < total; i++) {
    const d = document.createElement("span");
    d.className = `dot pop ${kind}`;
    container.appendChild(d);
  }
  container.querySelectorAll(".dot").forEach((d, i) => {
    const shouldFix = i < fixed;
    if (shouldFix && !d.classList.contains("fixed")) {
      d.classList.add("fixed", "just-fixed");
      d.addEventListener("animationend", () => d.classList.remove("just-fixed"), { once: true });
    } else if (!shouldFix) {
      d.classList.remove("fixed");
    }
  });
  container.setAttribute("aria-label", `${total} learners, ${fixed} fixed`);
}

// Count a number up (or down) to its new value.
function countTo(el, target, ms = 600) {
  const start = Number(el.textContent) || 0;
  if (start === target || reducedMotion()) { el.textContent = target; return; }
  const t0 = performance.now();
  (function frame(now) {
    const p = Math.min(1, (now - t0) / ms);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + (target - start) * eased);
    if (p < 1) requestAnimationFrame(frame);
  })(t0);
}

// Fade sections in as they scroll into view (once each).
function revealOnScroll(selector = ".on-scroll") {
  const els = document.querySelectorAll(selector);
  if (!("IntersectionObserver" in window) || reducedMotion()) {
    els.forEach(el => el.classList.add("seen"));
    return;
  }
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("seen"); io.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  els.forEach(el => io.observe(el));
}

// Give each element a stagger index for the .reveal animation.
function stagger(root, selector = ".reveal") {
  root.querySelectorAll(selector).forEach((el, i) => el.style.setProperty("--i", Math.min(i, 12)));
}
