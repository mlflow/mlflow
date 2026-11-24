export function getWindowTop() {
  try {
    // Accessing window.top properties may throw a security exception due to same origin policy
    // eslint-disable-next-line @databricks/no-window-top
    const top = window.top;
    // eslint-disable-next-line @typescript-eslint/no-unused-expressions
    top?.document;
    return top;
  } catch (e) {
    return null;
  }
}
