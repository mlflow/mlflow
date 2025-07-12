import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUserCircleIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.25 6.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M8 5.5A1.25 1.25 0 1 0 8 8a1.25 1.25 0 0 0 0-2.5", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 0 0-4.773 10.912A8.73 8.73 0 0 1 8 11c1.76 0 3.4.52 4.773 1.412A6.5 6.5 0 0 0 8 1.5m3.568 11.934A7.23 7.23 0 0 0 8 12.5a7.23 7.23 0 0 0-3.568.934A6.47 6.47 0 0 0 8 14.5a6.47 6.47 0 0 0 3.568-1.066", clipRule: "evenodd" })] }));
}
const UserCircleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUserCircleIcon });
});
UserCircleIcon.displayName = 'UserCircleIcon';
export default UserCircleIcon;
//# sourceMappingURL=UserCircleIcon.js.map