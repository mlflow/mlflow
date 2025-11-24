import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgHistoryIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsxs("g", { fill: "currentColor", clipPath: "url(#HistoryIcon_svg__a)", children: [_jsx("path", { d: "m3.507 7.73.963-.962 1.06 1.06-2.732 2.732L-.03 7.732l1.06-1.06.979.978a7 7 0 1 1 2.041 5.3l1.061-1.06a5.5 5.5 0 1 0-1.604-4.158" }), _jsx("path", { d: "M8.25 8V4h1.5v3.69l1.78 1.78-1.06 1.06-2-2A.75.75 0 0 1 8.25 8" })] }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const HistoryIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgHistoryIcon });
});
HistoryIcon.displayName = 'HistoryIcon';
export default HistoryIcon;
//# sourceMappingURL=HistoryIcon.js.map