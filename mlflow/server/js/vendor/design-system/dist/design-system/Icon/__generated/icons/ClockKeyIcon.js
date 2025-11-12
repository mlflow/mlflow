import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgClockKeyIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M8 1.5a6.5 6.5 0 0 0-5.07 10.57l-1.065 1.065A8 8 0 1 1 15.418 11h-1.65A6.5 6.5 0 0 0 8 1.5" }), _jsx("path", { fill: "currentColor", d: "M7.25 8V4h1.5v3.25H11v1.5H8A.75.75 0 0 1 7.25 8" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4 13a3 3 0 0 1 5.959-.5h4.291a.75.75 0 0 1 .75.75V16h-1.5v-2h-1v2H11v-2H9.83A3.001 3.001 0 0 1 4 13m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3", clipRule: "evenodd" })] }));
}
const ClockKeyIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgClockKeyIcon });
});
ClockKeyIcon.displayName = 'ClockKeyIcon';
export default ClockKeyIcon;
//# sourceMappingURL=ClockKeyIcon.js.map