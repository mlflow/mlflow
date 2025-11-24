import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgShieldIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 1.75A.75.75 0 0 1 2.75 1h10.5a.75.75 0 0 1 .75.75v7.465a5.75 5.75 0 0 1-2.723 4.889l-2.882 1.784a.75.75 0 0 1-.79 0l-2.882-1.784A5.75 5.75 0 0 1 2 9.214zm1.5.75V7h3.75V2.5zm5.25 0V7h3.75V2.5zm3.75 6H8.75v5.404l1.737-1.076A4.25 4.25 0 0 0 12.5 9.215zm-5.25 5.404V8.5H3.5v.715a4.25 4.25 0 0 0 2.013 3.613z", clipRule: "evenodd" }) }));
}
const ShieldIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgShieldIcon });
});
ShieldIcon.displayName = 'ShieldIcon';
export default ShieldIcon;
//# sourceMappingURL=ShieldIcon.js.map