import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDangerFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m15.78 11.533-4.242 4.243a.75.75 0 0 1-.53.22H4.996a.75.75 0 0 1-.53-.22L.224 11.533a.75.75 0 0 1-.22-.53v-6.01a.75.75 0 0 1 .22-.53L4.467.22a.75.75 0 0 1 .53-.22h6.01a.75.75 0 0 1 .53.22l4.243 4.242c.141.141.22.332.22.53v6.011a.75.75 0 0 1-.22.53m-8.528-.785a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0m1.5-5.75v4h-1.5v-4z", clipRule: "evenodd" }) }));
}
const DangerFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDangerFillIcon });
});
DangerFillIcon.displayName = 'DangerFillIcon';
export default DangerFillIcon;
//# sourceMappingURL=DangerFillIcon.js.map