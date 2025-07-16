import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLockShareIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M13.962 6.513a3.24 3.24 0 0 0-2.057.987H3.5v6.95H8v1.5H2.75A.75.75 0 0 1 2 15.2V6.75A.75.75 0 0 1 2.75 6H4V4a4 4 0 1 1 8 0v2h1.25a.75.75 0 0 1 .712.513M10.5 4v2h-5V4a2.5 2.5 0 0 1 5 0", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M11.5 12.036v-.072l1.671-.836a1.75 1.75 0 1 0-.67-1.342l-1.672.836a1.75 1.75 0 1 0 0 2.756l1.671.836v.036a1.75 1.75 0 1 0 .671-1.378z" })] }));
}
const LockShareIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLockShareIcon });
});
LockShareIcon.displayName = 'LockShareIcon';
export default LockShareIcon;
//# sourceMappingURL=LockShareIcon.js.map