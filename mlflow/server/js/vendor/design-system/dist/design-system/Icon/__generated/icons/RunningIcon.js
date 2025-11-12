import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgRunningIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { clipPath: "url(#RunningIcon_svg__a)", children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 1.5A6.5 6.5 0 0 0 1.5 8H0a8 8 0 0 1 8-8zm0 13A6.5 6.5 0 0 0 14.5 8H16a8 8 0 0 1-8 8z", clipRule: "evenodd" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const RunningIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgRunningIcon });
});
RunningIcon.displayName = 'RunningIcon';
export default RunningIcon;
//# sourceMappingURL=RunningIcon.js.map