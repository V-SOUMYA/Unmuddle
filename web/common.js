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

// One dot per learner. `fixed` of them are teal, the rest ochre.
function dots(total, fixed = 0, extraClass = "") {
  let html = `<div class="dots" role="img" aria-label="${total} learners, ${fixed} fixed">`;
  for (let i = 0; i < total; i++) {
    html += `<span class="dot ${i < fixed ? "fixed" : ""} ${extraClass}"></span>`;
  }
  return html + "</div>";
}
