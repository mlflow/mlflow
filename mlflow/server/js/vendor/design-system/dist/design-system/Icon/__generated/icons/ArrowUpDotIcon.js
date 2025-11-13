import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgArrowUpDotIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M8 1a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M12.53 9.47 8 4.94 3.47 9.47l1.06 1.06 2.72-2.72V15h1.5V7.81l2.72 2.72z" }) }));
}
const ArrowUpDotIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgArrowUpDotIcon });
});
ArrowUpDotIcon.displayName = 'ArrowUpDotIcon';
export default ArrowUpDotIcon;
//# sourceMappingURL=ArrowUpDotIcon.js.map