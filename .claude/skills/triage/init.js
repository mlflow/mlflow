// Init script for every agent-browser session during triage: pass it with --init-script on the
// `open` that starts the session.

// Report the browser as online. The triage sandbox has no network interface other than
// loopback, so Chromium sets navigator.onLine to false, and React Query then pauses requests
// that change data.
Object.defineProperty(Navigator.prototype, "onLine", { get: () => true, configurable: true });

// Defines caption(text, theme) for videos: shows text in a bar at the bottom of the page; caption() hides it.
// Pass theme "dark" on dark pages to use a light caption.
window.caption = (text, theme = "light") => {
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
  const dark = theme === "dark";
  el.style.background = dark ? "rgba(255,255,255,.85)" : "rgba(0,0,0,.6)";
  el.style.color = dark ? "#111" : "#fff";
  el.style.borderColor = dark ? "rgba(0,0,0,.2)" : "rgba(255,255,255,.4)";
  el.textContent = text || "";
  el.style.display = text ? "" : "none";
};
