import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgInfoIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 10.5v-3h1.5v3zM8 5a.75.75 0 1 1 0 1.5A.75.75 0 0 1 8 5" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12m0-1.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9", clipRule: "evenodd" })] }));
}
const InfoIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgInfoIcon });
});
InfoIcon.displayName = 'InfoIcon';
export default InfoIcon;
//# sourceMappingURL=InfoIcon.js.map