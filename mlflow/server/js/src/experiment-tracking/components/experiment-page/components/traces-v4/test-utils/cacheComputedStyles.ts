import { afterAll, beforeAll } from '@jest/globals';

/**
 * jsdom's `getComputedStyle` matches every stylesheet rule against the element and again for each
 * ancestor, and role queries by name call it for every descendant of every candidate. Reuse each
 * element's computed style until the document changes, so full-table suites stay within CI limits.
 */
export const cacheComputedStylesUntilDomChanges = () => {
  const { getComputedStyle } = window;
  let observer: MutationObserver | undefined;

  beforeAll(() => {
    const domChanges = new MutationObserver(() => {});
    domChanges.observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
    observer = domChanges;
    let styles = new WeakMap<Element, CSSStyleDeclaration>();
    // shortcut: a rule inserted into an existing sheet without a DOM change is not seen; Emotion only does that in production.
    let sheetCount = document.styleSheets.length;
    window.getComputedStyle = (element, pseudoElement) => {
      if (pseudoElement) {
        return getComputedStyle.call(window, element, pseudoElement);
      }
      if (domChanges.takeRecords().length > 0 || document.styleSheets.length !== sheetCount) {
        styles = new WeakMap();
        sheetCount = document.styleSheets.length;
      }
      let style = styles.get(element);
      if (!style) {
        style = getComputedStyle.call(window, element);
        styles.set(element, style);
      }
      return style;
    };
  });

  afterAll(() => {
    observer?.disconnect();
    window.getComputedStyle = getComputedStyle;
  });
};
