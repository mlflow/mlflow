import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarsDescendingVerticalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M7 12.75H1v-1.5h6zM15 4.75H1v-1.5h14zM1 7.25h10v1.5H1z" }) }));
}
const BarsDescendingVerticalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarsDescendingVerticalIcon });
});
BarsDescendingVerticalIcon.displayName = 'BarsDescendingVerticalIcon';
export default BarsDescendingVerticalIcon;
//# sourceMappingURL=BarsDescendingVerticalIcon.js.map