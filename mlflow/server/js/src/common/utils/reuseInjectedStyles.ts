/** The options `rc-util`'s `updateCSS` takes that decide which node an injection writes to. */
interface InjectOptions {
  attachTo?: Element | ShadowRoot;
  csp?: { nonce?: string };
  prepend?: boolean | 'queue';
  mark?: string;
}

/** Structural, so this does not import from `rc-util`, which is not a dependency of our own. */
export interface DynamicCSS {
  updateCSS: (css: string, key: string, option?: InjectOptions) => HTMLStyleElement;
}

/**
 * Skips a style injection that is already in place, for `rc-util`'s dynamic stylesheet module.
 *
 * Ant Design's icons inject their stylesheet from an effect in every icon that mounts, always
 * under the same key and with the same CSS. `updateCSS` finds the node it already wrote by
 * listing `document.head`'s children and reading an attribute off each `<style>` -- several
 * hundred of them in a suite of any length, all but one of them Emotion's one-tag-per-rule dev
 * output -- so every icon after the first pays for a scan that can only find what is there.
 *
 * The stylesheet is left as it would have been. The first injection under a key runs untouched,
 * and a repeat is skipped only while it is identical and the node it wrote is still in the
 * document, so what the rules compute to never changes.
 */
export const reuseInjectedStyles = <T extends DynamicCSS>(actual: T): T => {
  const injected = new Map<string, { css: string; node: HTMLStyleElement }>();
  // The mark names the attribute the key is written to, so one key under two marks is two nodes.
  const cacheKey = (key: string, option?: InjectOptions) => `${option?.mark ?? ''}|${key}`;

  const updateCSS = (css: string, key: string, option?: InjectOptions) => {
    // A container of its own or a nonce to reconcile makes `updateCSS` do more than write the
    // node, and neither arises from an icon mount.
    if (option?.attachTo || option?.csp?.nonce) {
      return actual.updateCSS(css, key, option);
    }
    const reusable = injected.get(cacheKey(key, option));
    // Asking the node whether it is still attached is what the scan was for. One that something
    // else removed -- including by `removeCSS`, which detaches it -- has to be injected again.
    if (reusable && reusable.css === css && reusable.node.parentNode) {
      return reusable.node;
    }
    const node = actual.updateCSS(css, key, option);
    injected.set(cacheKey(key, option), { css, node });
    return node;
  };

  return { ...actual, updateCSS };
};
