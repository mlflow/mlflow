import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDashboardIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5 8.75v3h4.75v-3zm0-1.5h4.75V2.5H2.5zm6.25-6.5v3h4.75v-3zm0 11V7h4.75v6.5z", clipRule: "evenodd" }) }));
}
const DashboardIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDashboardIcon });
});
DashboardIcon.displayName = 'DashboardIcon';
export default DashboardIcon;
//# sourceMappingURL=DashboardIcon.js.map