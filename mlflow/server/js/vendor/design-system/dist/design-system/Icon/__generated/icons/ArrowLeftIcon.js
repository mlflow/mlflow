import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgArrowLeftIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 8.03 8.03 1l1.061 1.06-5.22 5.22h11.19v1.5H3.87L9.091 14l-1.06 1.06z", clipRule: "evenodd" }) }));
}
const ArrowLeftIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgArrowLeftIcon });
});
ArrowLeftIcon.displayName = 'ArrowLeftIcon';
export default ArrowLeftIcon;
//# sourceMappingURL=ArrowLeftIcon.js.map