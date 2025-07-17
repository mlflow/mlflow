import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgArrowDownDotIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M8 15a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3M3.47 6.53 8 11.06l4.53-4.53-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47z" }) }));
}
const ArrowDownDotIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgArrowDownDotIcon });
});
ArrowDownDotIcon.displayName = 'ArrowDownDotIcon';
export default ArrowDownDotIcon;
//# sourceMappingURL=ArrowDownDotIcon.js.map