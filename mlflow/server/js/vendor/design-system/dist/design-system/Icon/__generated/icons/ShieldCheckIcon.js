import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgShieldCheckIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75v6.715a4.25 4.25 0 0 0 2.013 3.613L8 14.368l2.487-1.54A4.25 4.25 0 0 0 12.5 9.215V2.5zm6.22 2.97 1.06 1.06-3.53 3.53-2.03-2.03 1.06-1.06.97.97z", clipRule: "evenodd" }) }));
}
const ShieldCheckIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgShieldCheckIcon });
});
ShieldCheckIcon.displayName = 'ShieldCheckIcon';
export default ShieldCheckIcon;
//# sourceMappingURL=ShieldCheckIcon.js.map