import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronUpIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 7.083 5.053 10 4 8.958 8 5l4 3.958L10.947 10z", clipRule: "evenodd" }) }));
}
const ChevronUpIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronUpIcon });
});
ChevronUpIcon.displayName = 'ChevronUpIcon';
export default ChevronUpIcon;
//# sourceMappingURL=ChevronUpIcon.js.map