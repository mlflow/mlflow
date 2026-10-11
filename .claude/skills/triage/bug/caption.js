// Defines caption(text) for videos: shows text in a bar at the bottom of the page; caption() hides it.
window.caption = (text) => {
  let el = document.getElementById("triage-caption");
  if (!el) {
    el = document.createElement("div");
    el.id = "triage-caption";
    el.style.cssText =
      "position:fixed;left:50%;bottom:32px;transform:translateX(-50%);z-index:2147483647;" +
      "padding:10px 20px;border-radius:8px;backdrop-filter:blur(6px);" +
      "border:1px solid;box-shadow:0 4px 16px rgba(0,0,0,.4);" +
      "font:600 20px/1.3 system-ui,sans-serif;pointer-events:none;width:max-content;max-width:90vw;text-align:center";
    document.documentElement.appendChild(el);
  }
  // Invert the colors on dark pages so the caption stands out from the background.
  const [r, g, b, a = 1] = getComputedStyle(document.body)
    .backgroundColor.match(/[\d.]+/g)
    .map(Number);
  const dark = a > 0 && 0.299 * r + 0.587 * g + 0.114 * b < 128;
  el.style.background = dark ? "rgba(255,255,255,.85)" : "rgba(0,0,0,.6)";
  el.style.color = dark ? "#111" : "#fff";
  el.style.borderColor = dark ? "rgba(0,0,0,.2)" : "rgba(255,255,255,.4)";
  el.textContent = text || "";
  el.style.display = text ? "" : "none";
};
