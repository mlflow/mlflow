// Ensure React Query sends requests even if Chromium reports that the runner is offline.
Object.defineProperty(Navigator.prototype, "onLine", { get: () => true, configurable: true });

// Show a short explanation in recordings without changing the UI's layout.
window.caption = (text, theme = "light", position = "top") => {
  let el = document.getElementById("verify-caption");
  if (!el) {
    el = document.createElement("div");
    el.id = "verify-caption";
    el.style.cssText =
      "position:fixed;left:50%;transform:translateX(-50%);z-index:2147483647;" +
      "padding:10px 20px;border-radius:8px;backdrop-filter:blur(6px);" +
      "border:1px solid;box-shadow:0 4px 16px rgba(0,0,0,.4);" +
      "font:600 20px/1.3 system-ui,sans-serif;pointer-events:none;width:max-content;max-width:90vw;text-align:center";
    document.documentElement.appendChild(el);
  }
  const dark = theme === "dark";
  el.style.top = position === "bottom" ? "" : "32px";
  el.style.bottom = position === "bottom" ? "32px" : "";
  el.style.background = dark ? "rgba(255,255,255,.85)" : "rgba(0,0,0,.6)";
  el.style.color = dark ? "#111" : "#fff";
  el.style.borderColor = dark ? "rgba(0,0,0,.2)" : "rgba(255,255,255,.4)";
  el.textContent = text || "";
  el.style.display = text ? "" : "none";
};
