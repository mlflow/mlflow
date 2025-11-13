import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgQueryIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsxs("g", { fill: "currentColor", clipPath: "url(#QueryIcon_svg__a)", children: [_jsx("path", { fillRule: "evenodd", d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16h-.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z", clipRule: "evenodd" }), _jsx("path", { d: "M5.53 9.97 8.56 13l-3.03 3.03-1.06-1.06L6.44 13l-1.97-1.97zM14 14.5H9V16h5z" })] }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const QueryIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgQueryIcon });
});
QueryIcon.displayName = 'QueryIcon';
export default QueryIcon;
//# sourceMappingURL=QueryIcon.js.map