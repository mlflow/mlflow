import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarsDescendingHorizontalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M12.75 9v6h-1.5V9zM4.75 1v14h-1.5V1zM7.25 15V5h1.5v10z" }) }));
}
const BarsDescendingHorizontalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarsDescendingHorizontalIcon });
});
BarsDescendingHorizontalIcon.displayName = 'BarsDescendingHorizontalIcon';
export default BarsDescendingHorizontalIcon;
//# sourceMappingURL=BarsDescendingHorizontalIcon.js.map