import { beforeEach, describe, expect, jest, test } from '@jest/globals';

import { type DynamicCSS, reuseInjectedStyles } from './reuseInjectedStyles';

// `removeCSS` is not patched, so it stands for the rest of the module the generic carries over.
type StubCSS = DynamicCSS & { removeCSS: (key: string) => void };

const stubActual = () => {
  const calls: { css: string; key: string }[] = [];
  const actual = {
    updateCSS: jest.fn((css: string, key: string) => {
      calls.push({ css, key });
      const node = document.createElement('style');
      node.innerHTML = css;
      node.setAttribute('rc-util-key', key);
      document.head.appendChild(node);
      return node;
    }),
    removeCSS: jest.fn((key: string) => {
      document.head.querySelectorAll(`[rc-util-key="${key}"]`).forEach((node) => node.remove());
    }),
    injectCSS: jest.fn(),
    clearContainerCache: jest.fn(),
  } as unknown as StubCSS;
  return { actual, calls };
};

describe('reuseInjectedStyles', () => {
  // The document is shared across the tests in a file, so each starts with a head of its own.
  beforeEach(() => {
    document.head.querySelectorAll('[rc-util-key]').forEach((node) => node.remove());
  });

  test('injects once and reuses that node for an identical repeat', () => {
    const { actual, calls } = stubActual();
    const { updateCSS } = reuseInjectedStyles(actual);

    const first = updateCSS('.anticon { display: inline-block; }', '@ant-design-icons');
    const second = updateCSS('.anticon { display: inline-block; }', '@ant-design-icons');

    expect(second).toBe(first);
    expect(calls).toHaveLength(1);
    // The stylesheet is what it would have been without any of this.
    expect(document.head.querySelectorAll('[rc-util-key]')).toHaveLength(1);
    expect(first.innerHTML).toBe('.anticon { display: inline-block; }');
  });

  test('passes through a changed stylesheet for the same key', () => {
    const { actual, calls } = stubActual();
    const { updateCSS } = reuseInjectedStyles(actual);

    updateCSS('.a { color: red; }', 'k');
    updateCSS('.a { color: blue; }', 'k');

    expect(calls.map(({ css }) => css)).toEqual(['.a { color: red; }', '.a { color: blue; }']);
  });

  test('injects again when the node it wrote was removed', () => {
    const { actual, calls } = stubActual();
    const { updateCSS } = reuseInjectedStyles(actual);

    updateCSS('.a {}', 'k').remove();
    updateCSS('.a {}', 'k');

    expect(calls).toHaveLength(2);
    expect(document.head.querySelectorAll('[rc-util-key="k"]')).toHaveLength(1);
  });

  test('injects again after the key is removed', () => {
    const { actual, calls } = stubActual();
    const patched = reuseInjectedStyles(actual);

    patched.updateCSS('.a {}', 'k');
    patched.removeCSS('k');
    patched.updateCSS('.a {}', 'k');

    expect(calls).toHaveLength(2);
  });

  test('keeps one key under two marks apart', () => {
    const { actual, calls } = stubActual();
    const { updateCSS } = reuseInjectedStyles(actual);

    updateCSS('.a {}', 'k', { mark: 'one' });
    updateCSS('.a {}', 'k', { mark: 'two' });

    expect(calls).toHaveLength(2);
  });

  test('defers to the real call for its own container or a nonce', () => {
    const { actual, calls } = stubActual();
    const { updateCSS } = reuseInjectedStyles(actual);
    const container = document.createElement('div');

    updateCSS('.a {}', 'k', { attachTo: container });
    updateCSS('.a {}', 'k', { attachTo: container });
    updateCSS('.b {}', 'n', { csp: { nonce: 'abc' } });
    updateCSS('.b {}', 'n', { csp: { nonce: 'abc' } });

    expect(calls).toHaveLength(4);
  });
});
