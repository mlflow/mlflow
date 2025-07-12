import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUserGroupIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2.25 3.75a2.75 2.75 0 1 1 5.5 0 2.75 2.75 0 0 1-5.5 0M5 2.5A1.25 1.25 0 1 0 5 5a1.25 1.25 0 0 0 0-2.5M9.502 14H.75a.75.75 0 0 1-.75-.75V11a.75.75 0 0 1 .164-.469C1.298 9.115 3.077 8 5.125 8c1.76 0 3.32.822 4.443 1.952A5.55 5.55 0 0 1 11.75 9.5c1.642 0 3.094.745 4.041 1.73a.75.75 0 0 1 .209.52v1.5a.75.75 0 0 1-.75.75zM1.5 12.5v-1.228C2.414 10.228 3.72 9.5 5.125 9.5c1.406 0 2.71.728 3.625 1.772V12.5zm8.75 0h4.25v-.432A4.17 4.17 0 0 0 11.75 11c-.53 0-1.037.108-1.5.293zM11.75 3.5a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M11 5.75a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0", clipRule: "evenodd" }) }));
}
const UserGroupIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUserGroupIcon });
});
UserGroupIcon.displayName = 'UserGroupIcon';
export default UserGroupIcon;
//# sourceMappingURL=UserGroupIcon.js.map