import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgZoomOutIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 17", ...props, children: [_jsx("path", { fill: "currentColor", d: "M11 7.25H5v1.5h6z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 8a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 1 8m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11", clipRule: "evenodd" })] }));
}
const ZoomOutIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgZoomOutIcon });
});
ZoomOutIcon.displayName = 'ZoomOutIcon';
export default ZoomOutIcon;
//# sourceMappingURL=ZoomOutIcon.js.map