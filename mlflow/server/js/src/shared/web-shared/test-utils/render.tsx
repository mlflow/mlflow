/**
 * A thin wrapper around `@testing-library/react` used by model-trace-explorer/v2
 * tests. It re-exports RTL's pure API and a custom `render` that disables Emotion
 * CSS by default (for JSDOM performance) and peels `DesignSystemProvider`'s
 * layout-neutral scope wrapper off `container`.
 */
import createCache from '@emotion/cache';
import { CacheProvider } from '@emotion/react';
// eslint-disable-next-line no-restricted-imports
import type { RenderOptions } from '@testing-library/react';
// eslint-disable-next-line no-restricted-imports
import { render as rtlRender } from '@testing-library/react';
import type { ReactNode } from 'react';
import React from 'react';

export * from '@testing-library/react/pure';

// Keep in sync with DS_OVERRIDE_TOKENS_WRAPPER_TESTID exported from '@databricks/design-system'
// (DesignSystemProvider). Duplicated as a literal so this foundational test util keeps its zero
// runtime dependencies rather than taking a dependency on design-system just for the constant.
const DS_OVERRIDE_TOKENS_WRAPPER_TESTID = 'ds-override-tokens-wrapper';

const noopCache = createCache({
  key: 'jest-css',
  container: {
    insertBefore: () => {},
  } as any, // dummy construction for not inserting
});

function NoopCssCacheProvider({ children }: { children: ReactNode }) {
  return <CacheProvider value={noopCache}>{children}</CacheProvider>;
}

export interface CustomRenderOptions extends RenderOptions {
  /**
   * Set to `true` to enable CSS rendering from `@emotion/react` package.
   * CSS heavily affects the performance of JSDOM so disabling Emotion
   * improves the performance of tests. Tests that assert on CSS styling
   * of the rendered component must set this option to `true`.
   */
  enableEmotionCSS?: boolean;
}

export function render(
  ui: ReactNode,
  { enableEmotionCSS = false, ...options }: CustomRenderOptions = {},
): ReturnType<typeof rtlRender> {
  const result = rtlRender(ui, {
    ...options,
    wrapper: composeWrappers([options.wrapper, enableEmotionCSS ? undefined : NoopCssCacheProvider]),
  });
  const container = peelDesignSystemScopeWrappers(result.container);
  return {
    ...result,
    container,
    // Base `asFragment` on the peeled `container` so it captures the component's own output without
    // the scope wrapper(s) — matching what it produced before the wrapper existed (RTL's default
    // reads the outer container, which would include the wrapper).
    asFragment: () =>
      typeof document.createRange === 'function'
        ? document.createRange().createContextualFragment(container.innerHTML)
        : (() => {
            const template = document.createElement('template');
            template.innerHTML = container.innerHTML;
            return template.content;
          })(),
  };
}

/**
 * Returns the element holding the rendered component's own output. `DesignSystemProvider` (supplied by
 * the component-under-test or its test wrapper) renders a layout-neutral (`display: contents`) scope
 * wrapper around its children, and nested providers (notebook theme, notifications, …) legitimately
 * stack more. Descending through every consecutive scope wrapper keeps child-scoped queries and
 * emptiness assertions (`toBeEmptyDOMElement()`, `container.firstChild`) reading exactly what they did
 * before the wrapper existed. Keyed on the wrapper testid, so it only ever walks DSP scope wrappers;
 * a render with no `DesignSystemProvider` gets its container back unchanged.
 */
export function peelDesignSystemScopeWrappers(container: HTMLElement): HTMLElement {
  let innermostContainer = container;
  while (
    innermostContainer.childNodes.length === 1 &&
    innermostContainer.firstChild instanceof HTMLElement &&
    innermostContainer.firstChild.dataset['testid'] === DS_OVERRIDE_TOKENS_WRAPPER_TESTID
  ) {
    innermostContainer = innermostContainer.firstChild;
  }
  return innermostContainer;
}

/**
 * Compose a list of React elements into a single element. If an element is
 * undefined, it is ignored. The last element in the list is the outermost
 * element.
 */
function composeWrappers(wrappers: Array<RenderOptions['wrapper']>) {
  function AllWrappers({ children }: { children: ReactNode }) {
    return wrappers.reduce((acc, Wrapper) => (Wrapper ? <Wrapper>{acc}</Wrapper> : acc), <>{children}</>);
  }
  return AllWrappers;
}
