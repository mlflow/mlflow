import { afterEach, describe, expect, it } from '@jest/globals';

import { shouldPreventDrawerDismiss } from './AssistantAwareDrawer';

// These tests exercise the pure guard directly against real DOM nodes so they are fully
// deterministic: no component rendering, no Radix DismissableLayer event simulation, and no
// timers/async that could make the assertions flaky.
describe('shouldPreventDrawerDismiss', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  const renderTarget = (html: string): HTMLElement => {
    const container = document.createElement('div');
    container.innerHTML = html;
    document.body.appendChild(container);
    return container.querySelector<HTMLElement>('[data-target]')!;
  };

  it('prevents dismissal for a portaled Radix popover/menu (e.g. the provider picker)', () => {
    const target = renderTarget(
      '<div data-radix-popper-content-wrapper><div role="menu"><span data-target>Claude</span></div></div>',
    );
    expect(shouldPreventDrawerDismiss(target)).toBe(true);
  });

  it('prevents dismissal for assistant UI', () => {
    const target = renderTarget('<div data-assistant-ui="true"><button data-target>Ask</button></div>');
    expect(shouldPreventDrawerDismiss(target)).toBe(true);
  });

  it('prevents dismissal for the drawer resize handle', () => {
    const target = renderTarget('<div data-drawer-resize-handle="true" data-target></div>');
    expect(shouldPreventDrawerDismiss(target)).toBe(true);
  });

  it('allows dismissal for an unrelated element outside the drawer', () => {
    const target = renderTarget('<div><span data-target>elsewhere</span></div>');
    expect(shouldPreventDrawerDismiss(target)).toBe(false);
  });

  it('allows dismissal when there is no target', () => {
    expect(shouldPreventDrawerDismiss(null)).toBe(false);
  });
});
