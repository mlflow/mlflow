import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarsAscendingHorizontalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M3.25 9v6h1.5V9zM11.25 1v14h1.5V1zM8.75 15V5h-1.5v10z" }) }));
}
const BarsAscendingHorizontalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarsAscendingHorizontalIcon });
});
BarsAscendingHorizontalIcon.displayName = 'BarsAscendingHorizontalIcon';
export default BarsAscendingHorizontalIcon;
//# sourceMappingURL=BarsAscendingHorizontalIcon.js.map