import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCatalogSharedIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.5 5c-.356 0-.694-.074-1-.208v8.458c0 .69.56 1.25 1.25 1.25H10V16H4.75A2.75 2.75 0 0 1 2 13.25V2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75zm0-1.5a1 1 0 0 1 0-2h8v2z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M14 6.5a2 2 0 0 0-1.953 2.433l-.944.648a2 2 0 1 0 .105 3.262l.858.644a2 2 0 1 0 .9-1.2l-.988-.74a2 2 0 0 0-.025-.73l.944-.649A2 2 0 1 0 14 6.5m-.5 2a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0m-4 2.75a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0M14 13.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1", clipRule: "evenodd" })] }));
}
const CatalogSharedIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCatalogSharedIcon });
});
CatalogSharedIcon.displayName = 'CatalogSharedIcon';
export default CatalogSharedIcon;
//# sourceMappingURL=CatalogSharedIcon.js.map