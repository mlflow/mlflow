import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUserBadgeIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 5.25a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5M6.75 8a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m4.401 2.5.386-.867A2.75 2.75 0 0 1 7.3 0h1.4a2.75 2.75 0 0 1 2.513 1.633l.386.867h1.651a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75v-12a.75.75 0 0 1 .75-.75zm1.756-.258A1.25 1.25 0 0 1 7.3 1.5h1.4c.494 0 .942.29 1.143.742l.114.258H6.043zM8 12a8.7 8.7 0 0 0-4.5 1.244V4h9v9.244A8.7 8.7 0 0 0 8 12m0 1.5c1.342 0 2.599.364 3.677 1H4.323A7.2 7.2 0 0 1 8 13.5", clipRule: "evenodd" })] }));
}
const UserBadgeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUserBadgeIcon });
});
UserBadgeIcon.displayName = 'UserBadgeIcon';
export default UserBadgeIcon;
//# sourceMappingURL=UserBadgeIcon.js.map