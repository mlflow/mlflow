// eslint-disable-next-line import/no-extraneous-dependencies
import createCache from '@emotion/cache';
import { CacheProvider } from '@emotion/react';
import React, { useEffect, useRef, useState } from 'react';
// eslint-disable-next-line import/no-extraneous-dependencies
import ReactShadowRoot from 'react-shadow-root';

import { ApplyDesignSystemFlags, DesignSystemProvider } from '../DesignSystemProvider';
import { useDesignSystemFlags } from '../Hooks';

export const ShadowDomWrapper: React.FC = ({ children }) => {
  const popupContainer = useRef<HTMLDivElement>(null);
  const stylesContainer = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);
  const flags = useDesignSystemFlags();
  const emotionCache = createCache({
    key: 'design-system-css',
    container: stylesContainer.current || document.body,
  });

  // Hack to force re-render to populate refs, which the `DesignSystemProvider` needs
  // on the first render.
  useEffect(() => {
    setTimeout(() => {
      setLoaded(true);
    }, 0);
  }, [popupContainer, stylesContainer]);

  return (
    <div>
      <ReactShadowRoot id="shadow-test-container" mode="open">
        <link rel="stylesheet" href="/index.css" />
        <link rel="stylesheet" href="/index-dark.css" />

        <div ref={popupContainer} />
        {loaded && popupContainer.current && (
          <CacheProvider value={emotionCache}>
            <DesignSystemProvider getPopupContainer={() => popupContainer.current || document.body}>
              <ApplyDesignSystemFlags flags={flags}>{children}</ApplyDesignSystemFlags>
            </DesignSystemProvider>
          </CacheProvider>
        )}
        <div ref={stylesContainer || null} />
      </ReactShadowRoot>
    </div>
  );
};
