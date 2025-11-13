import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgZoomMarqueeSelection(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M1 1.75V4h1.5V2.5H4V1H1.75a.75.75 0 0 0-.75.75M14.25 1H12v1.5h1.5V4H15V1.75a.75.75 0 0 0-.75-.75M4 15H1.75a.75.75 0 0 1-.75-.75V12h1.5v1.5H4zM6 2.5h4V1H6zM1 10V6h1.5v4z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.053 9.27a5.217 5.217 0 1 1 9.397 3.122l1.69 1.69-1.062 1.06-1.688-1.69A5.217 5.217 0 0 1 4.053 9.27M9.27 5.555a3.717 3.717 0 1 0 0 7.434 3.717 3.717 0 0 0 0-7.434", clipRule: "evenodd" })] }));
}
const ZoomMarqueeSelection = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgZoomMarqueeSelection });
});
ZoomMarqueeSelection.displayName = 'ZoomMarqueeSelection';
export default ZoomMarqueeSelection;
//# sourceMappingURL=ZoomMarqueeSelection.js.map