import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronRightIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8.917 8 6 5.053 7.042 4 11 8l-3.958 4L6 10.947z", clipRule: "evenodd" }) }));
}
const ChevronRightIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronRightIcon });
});
ChevronRightIcon.displayName = 'ChevronRightIcon';
export default ChevronRightIcon;
//# sourceMappingURL=ChevronRightIcon.js.map