import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarsAscendingVerticalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M7 3.25H1v1.5h6zM15 11.25H1v1.5h14zM1 8.75h10v-1.5H1z" }) }));
}
const BarsAscendingVerticalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarsAscendingVerticalIcon });
});
BarsAscendingVerticalIcon.displayName = 'BarsAscendingVerticalIcon';
export default BarsAscendingVerticalIcon;
//# sourceMappingURL=BarsAscendingVerticalIcon.js.map