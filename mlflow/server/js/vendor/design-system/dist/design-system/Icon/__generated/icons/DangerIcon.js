import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDangerIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.248 10.748a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0M8.748 4.998v4h-1.5v-4z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m11.533 15.776 4.243-4.243a.75.75 0 0 0 .22-.53v-6.01a.75.75 0 0 0-.22-.53L11.533.22a.75.75 0 0 0-.53-.22h-6.01a.75.75 0 0 0-.53.22L.22 4.462a.75.75 0 0 0-.22.53v6.011c0 .199.079.39.22.53l4.242 4.243c.141.14.332.22.53.22h6.011a.75.75 0 0 0 .53-.22m2.963-10.473v5.39l-3.804 3.803H5.303L1.5 10.692V5.303L5.303 1.5h5.39z", clipRule: "evenodd" })] }));
}
const DangerIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDangerIcon });
});
DangerIcon.displayName = 'DangerIcon';
export default DangerIcon;
//# sourceMappingURL=DangerIcon.js.map