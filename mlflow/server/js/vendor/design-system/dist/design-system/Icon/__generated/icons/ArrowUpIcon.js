import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgArrowUpIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m8.03 1 7.03 7.03L14 9.091l-5.22-5.22v11.19h-1.5V3.87l-5.22 5.22L1 8.031z", clipRule: "evenodd" }) }));
}
const ArrowUpIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgArrowUpIcon });
});
ArrowUpIcon.displayName = 'ArrowUpIcon';
export default ArrowUpIcon;
//# sourceMappingURL=ArrowUpIcon.js.map