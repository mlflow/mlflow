import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChecklistIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "m5.5 2 1.06 1.06-3.53 3.531L1 4.561 2.06 3.5l.97.97zM15.03 4.53h-7v-1.5h7zM1.03 14.53v-1.5h14v1.5zM8.03 9.53h7v-1.5h-7zM6.56 8.06 5.5 7 3.03 9.47l-.97-.97L1 9.56l2.03 2.031z" }) }));
}
const ChecklistIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChecklistIcon });
});
ChecklistIcon.displayName = 'ChecklistIcon';
export default ChecklistIcon;
//# sourceMappingURL=ChecklistIcon.js.map