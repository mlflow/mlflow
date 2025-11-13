import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPinIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.75 0A.75.75 0 0 0 5 .75v4.007a2.25 2.25 0 0 1-.659 1.591L2.22 8.47A.75.75 0 0 0 2 9v2.25c0 .414.336.75.75.75h4.5v4h1.5v-4h4.5a.75.75 0 0 0 .75-.75V9a.75.75 0 0 0-.22-.53L11.66 6.348A2.25 2.25 0 0 1 11 4.758V.75a.75.75 0 0 0-.75-.75zm.75 4.757V1.5h3v3.257a3.75 3.75 0 0 0 1.098 2.652L12.5 9.311V10.5h-9V9.31L5.402 7.41A3.75 3.75 0 0 0 6.5 4.757", clipRule: "evenodd" }) }));
}
const PinIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPinIcon });
});
PinIcon.displayName = 'PinIcon';
export default PinIcon;
//# sourceMappingURL=PinIcon.js.map