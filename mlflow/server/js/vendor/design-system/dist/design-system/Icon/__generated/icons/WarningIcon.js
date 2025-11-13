import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgWarningIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 10V6.5h1.5V10zM8 12.5A.75.75 0 1 0 8 11a.75.75 0 0 0 0 1.5" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 1a.75.75 0 0 1 .649.374l7.25 12.5A.75.75 0 0 1 15.25 15H.75a.75.75 0 0 1-.649-1.126l7.25-12.5A.75.75 0 0 1 8 1m0 2.245L2.052 13.5h11.896z", clipRule: "evenodd" })] }));
}
const WarningIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgWarningIcon });
});
WarningIcon.displayName = 'WarningIcon';
export default WarningIcon;
//# sourceMappingURL=WarningIcon.js.map