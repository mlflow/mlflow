import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronDownIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 8.917 10.947 6 12 7.042 8 11 4 7.042 5.053 6z", clipRule: "evenodd" }) }));
}
const ChevronDownIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronDownIcon });
});
ChevronDownIcon.displayName = 'ChevronDownIcon';
export default ChevronDownIcon;
//# sourceMappingURL=ChevronDownIcon.js.map