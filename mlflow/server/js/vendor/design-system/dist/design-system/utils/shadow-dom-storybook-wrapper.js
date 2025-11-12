import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
// eslint-disable-next-line import/no-extraneous-dependencies
import createCache from '@emotion/cache';
import { CacheProvider } from '@emotion/react';
import { useEffect, useRef, useState } from 'react';
// eslint-disable-next-line import/no-extraneous-dependencies
import ReactShadowRoot from 'react-shadow-root';
import { ApplyDesignSystemFlags, DesignSystemProvider } from '../DesignSystemProvider';
import { useDesignSystemFlags } from '../Hooks';
export const ShadowDomWrapper = ({ children }) => {
    const popupContainer = useRef(null);
    const stylesContainer = useRef(null);
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
    return (_jsx("div", { children: _jsxs(ReactShadowRoot, { id: "shadow-test-container", mode: "open", children: [_jsx("link", { rel: "stylesheet", href: "/index.css" }), _jsx("link", { rel: "stylesheet", href: "/index-dark.css" }), _jsx("div", { ref: popupContainer }), loaded && popupContainer.current && (_jsx(CacheProvider, { value: emotionCache, children: _jsx(DesignSystemProvider, { getPopupContainer: () => popupContainer.current || document.body, children: _jsx(ApplyDesignSystemFlags, { flags: flags, children: children }) }) })), _jsx("div", { ref: stylesContainer || null })] }) }));
};
//# sourceMappingURL=shadow-dom-storybook-wrapper.js.map