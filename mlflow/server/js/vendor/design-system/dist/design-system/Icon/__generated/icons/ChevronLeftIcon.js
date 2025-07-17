import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronLeftIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.083 8 10 10.947 8.958 12 5 8l3.958-4L10 5.053z", clipRule: "evenodd" }) }));
}
const ChevronLeftIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronLeftIcon });
});
ChevronLeftIcon.displayName = 'ChevronLeftIcon';
export default ChevronLeftIcon;
//# sourceMappingURL=ChevronLeftIcon.js.map