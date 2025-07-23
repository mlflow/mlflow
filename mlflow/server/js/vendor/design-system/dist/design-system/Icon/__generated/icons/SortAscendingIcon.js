import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSortAscendingIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "m11.5.94 4.03 4.03-1.06 1.06-2.22-2.22V10h-1.5V3.81L8.53 6.03 7.47 4.97zM1 4.5h4V6H1zM1 12.5h10V14H1zM8 8.5H1V10h7z" }) }));
}
const SortAscendingIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSortAscendingIcon });
});
SortAscendingIcon.displayName = 'SortAscendingIcon';
export default SortAscendingIcon;
//# sourceMappingURL=SortAscendingIcon.js.map