import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgIndentIncreaseIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM1.97 6.03l1.06-1.06L6.06 8l-3.03 3.03-1.06-1.06L3.94 8z" }) }));
}
const IndentIncreaseIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgIndentIncreaseIcon });
});
IndentIncreaseIcon.displayName = 'IndentIncreaseIcon';
export default IndentIncreaseIcon;
//# sourceMappingURL=IndentIncreaseIcon.js.map