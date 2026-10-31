import { afterAll, beforeAll } from '@jest/globals';

/**
 * jsdom's `getComputedStyle` matches every stylesheet rule against the element and again for each
 * ancestor, and role queries by name call it for every descendant of every candidate. Reuse each
 * element's computed style until the document or a stylesheet changes, so suites that render large
 * design-system pages stay within CI limits.
 */
export const cacheComputedStylesUntilDomChanges = () => {
  const { getComputedStyle } = window;
  let restore = () => {};

  beforeAll(() => {
    let styles = new WeakMap<Element, CSSStyleDeclaration>();
    let stale = false;
    // Records reach the callback at the next microtask; `takeRecords()` catches those made since.
    const domChanges = new MutationObserver(() => {
      stale = true;
    });
    domChanges.observe(document, { subtree: true, childList: true, attributes: true, characterData: true });

    // Rules added with `insertRule` (as Emotion's speedy mode does) change styles without a DOM mutation.
    const probe = document.createElement('style');
    document.head.appendChild(probe);
    const sheetPrototype = Object.getPrototypeOf(probe.sheet);
    probe.remove();
    const { insertRule, deleteRule } = sheetPrototype;
    sheetPrototype.insertRule = function (...args: Parameters<CSSStyleSheet['insertRule']>) {
      stale = true;
      return insertRule.apply(this, args);
    };
    sheetPrototype.deleteRule = function (...args: Parameters<CSSStyleSheet['deleteRule']>) {
      stale = true;
      return deleteRule.apply(this, args);
    };

    // `:focus` and `:checked` rules follow state that changes without a DOM mutation. `focus` and `blur`
    // fire before `focusin` and `focusout`, so listeners on them see the new state too. Not seen: an input
    // `value` set from code without an event, `form.reset()`, and edits through `cssRules[i].style`.
    const markStale = () => {
      stale = true;
    };
    const stateEvents = ['focus', 'blur', 'focusin', 'focusout', 'input', 'change'];
    stateEvents.forEach((type) => document.addEventListener(type, markStale, true));
    const checked = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'checked');
    Object.defineProperty(HTMLInputElement.prototype, 'checked', {
      ...checked,
      set(value: boolean) {
        stale = true;
        checked?.set?.call(this, value);
      },
    });

    window.getComputedStyle = (element, pseudoElement) => {
      // The observer sees only this document's tree, so detached, shadow and other-document elements are not cached.
      if (pseudoElement || element.getRootNode() !== document) {
        return getComputedStyle.call(window, element, pseudoElement);
      }
      if (domChanges.takeRecords().length > 0 || stale) {
        styles = new WeakMap();
        stale = false;
      }
      let style = styles.get(element);
      if (!style) {
        style = getComputedStyle.call(window, element);
        styles.set(element, style);
      }
      return style;
    };

    restore = () => {
      domChanges.disconnect();
      sheetPrototype.insertRule = insertRule;
      sheetPrototype.deleteRule = deleteRule;
      stateEvents.forEach((type) => document.removeEventListener(type, markStale, true));
      if (checked) {
        Object.defineProperty(HTMLInputElement.prototype, 'checked', checked);
      }
      window.getComputedStyle = getComputedStyle;
    };
  });

  afterAll(() => restore());
};
