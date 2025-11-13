import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCheckCircleIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M11.53 6.53 7 11.06 4.47 8.53l1.06-1.06L7 8.94l3.47-3.47z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13", clipRule: "evenodd" })] }));
}
const CheckCircleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCheckCircleIcon });
});
CheckCircleIcon.displayName = 'CheckCircleIcon';
export default CheckCircleIcon;
//# sourceMappingURL=CheckCircleIcon.js.map