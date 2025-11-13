import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.75 4.5a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h2a3.5 3.5 0 1 0 0-7zM6.5 10V6h1.25a2 2 0 1 1 0 4z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z", clipRule: "evenodd" })] }));
}
const DIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDIcon });
});
DIcon.displayName = 'DIcon';
export default DIcon;
//# sourceMappingURL=DIcon.js.map