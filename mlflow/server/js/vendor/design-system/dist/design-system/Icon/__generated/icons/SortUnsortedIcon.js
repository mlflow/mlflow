import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSortUnsortedIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M11.5.94 7.47 4.97l1.06 1.06 2.22-2.22v8.38L8.53 9.97l-1.06 1.06 4.03 4.03 4.03-4.03-1.06-1.06-2.22 2.22V3.81l2.22 2.22 1.06-1.06zM6 3.5H1V5h5zM6 11.5H1V13h5zM1 7.5h5V9H1z" }) }));
}
const SortUnsortedIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSortUnsortedIcon });
});
SortUnsortedIcon.displayName = 'SortUnsortedIcon';
export default SortUnsortedIcon;
//# sourceMappingURL=SortUnsortedIcon.js.map