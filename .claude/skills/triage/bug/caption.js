// Defines caption(text) for videos: shows text in a bar at the bottom of the page; caption() hides it.
window.caption = (text) => {
  let el = document.getElementById("triage-caption");
  if (!el) {
    el = document.createElement("div");
    el.id = "triage-caption";
    el.style.cssText =
      "position:fixed;left:50%;bottom:32px;transform:translateX(-50%);z-index:2147483647;" +
      "padding:10px 20px;border-radius:8px;background:rgba(0,0,0,.6);backdrop-filter:blur(6px);" +
      "border:1px solid rgba(255,255,255,.4);box-shadow:0 4px 16px rgba(0,0,0,.4);color:#fff;" +
      "font:600 20px/1.3 system-ui,sans-serif;pointer-events:none;white-space:nowrap";
    document.documentElement.appendChild(el);
  }
  el.textContent = text || "";
  el.style.display = text ? "" : "none";
};
