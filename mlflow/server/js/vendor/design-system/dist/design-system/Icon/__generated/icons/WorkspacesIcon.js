import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgWorkspacesIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M2.5 1a.75.75 0 0 0-.75.75v3c0 .414.336.75.75.75H6V4H3.25V2.5h9.5V4H10v1.5h3.5a.75.75 0 0 0 .75-.75v-3A.75.75 0 0 0 13.5 1z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 12.25c0-1.26.848-2.322 2.004-2.648A2.75 2.75 0 0 1 4.75 7h2.5V4h1.5v3h2.5a2.75 2.75 0 0 1 2.746 2.602 2.751 2.751 0 1 1-3.371 3.47 2.751 2.751 0 0 1-5.25 0A2.751 2.751 0 0 1 0 12.25M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.625.428a2.76 2.76 0 0 0-1.867-1.822A1.25 1.25 0 0 1 4.75 8.5h2.5v1.104c-.892.252-1.6.942-1.875 1.824M8.75 9.604V8.5h2.5c.642 0 1.17.483 1.242 1.106a2.76 2.76 0 0 0-1.867 1.822A2.76 2.76 0 0 0 8.75 9.604M12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0", clipRule: "evenodd" })] }));
}
const WorkspacesIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgWorkspacesIcon });
});
WorkspacesIcon.displayName = 'WorkspacesIcon';
export default WorkspacesIcon;
//# sourceMappingURL=WorkspacesIcon.js.map