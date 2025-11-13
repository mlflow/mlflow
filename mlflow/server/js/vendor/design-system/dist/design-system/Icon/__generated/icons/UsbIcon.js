import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUsbIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M8 0a.75.75 0 0 1 .65.375l1.299 2.25a.75.75 0 0 1-.65 1.125H8.75V9.5h2.75V8h-.25a.75.75 0 0 1-.75-.75v-2a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75H13v2.25a.75.75 0 0 1-.75.75h-3.5v1.668a1.75 1.75 0 1 1-1.5 0V11h-3.5a.75.75 0 0 1-.75-.75V7.832a1.75 1.75 0 1 1 1.5 0V9.5h2.75V3.75h-.549a.75.75 0 0 1-.65-1.125l1.3-2.25A.75.75 0 0 1 8 0" }) }));
}
const UsbIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUsbIcon });
});
UsbIcon.displayName = 'UsbIcon';
export default UsbIcon;
//# sourceMappingURL=UsbIcon.js.map