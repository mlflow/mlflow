import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCatalogHIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3.5 13.25V4.792c.306.134.644.208 1 .208h8v3H14V.75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16H8.5v-1.5H4.75c-.69 0-1.25-.56-1.25-1.25m9-9.75h-8a1 1 0 0 1 0-2h8z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M10 9v7h1.5v-2.75h3V16H16V9h-1.5v2.75h-3V9z" })] }));
}
const CatalogHIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCatalogHIcon });
});
CatalogHIcon.displayName = 'CatalogHIcon';
export default CatalogHIcon;
//# sourceMappingURL=CatalogHIcon.js.map