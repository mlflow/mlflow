import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgShieldOffIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M13.378 11.817A5.75 5.75 0 0 0 14 9.215V1.75a.75.75 0 0 0-.75-.75H2.75a.8.8 0 0 0-.17.02L4.06 2.5h8.44v6.715c0 .507-.09 1.002-.26 1.464z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m1.97 2.53-1 1L2 4.56v4.655a5.75 5.75 0 0 0 2.723 4.889l2.882 1.784a.75.75 0 0 0 .79 0l2.882-1.784.162-.104 1.53 1.53 1-1zM3.5 9.215V6.06l6.852 6.851L8 14.368l-2.487-1.54A4.25 4.25 0 0 1 3.5 9.215", clipRule: "evenodd" })] }));
}
const ShieldOffIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgShieldOffIcon });
});
ShieldOffIcon.displayName = 'ShieldOffIcon';
export default ShieldOffIcon;
//# sourceMappingURL=ShieldOffIcon.js.map