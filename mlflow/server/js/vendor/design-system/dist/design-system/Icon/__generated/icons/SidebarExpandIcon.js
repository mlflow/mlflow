import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSidebarExpandIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H15v-1.5H5.5v-11H15V1zM4 2.5H2.5v11H4z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M11.19 8.75 9.97 9.97l1.06 1.06L14.06 8l-3.03-3.03-1.06 1.06 1.22 1.22H7v1.5z" })] }));
}
const SidebarExpandIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSidebarExpandIcon });
});
SidebarExpandIcon.displayName = 'SidebarExpandIcon';
export default SidebarExpandIcon;
//# sourceMappingURL=SidebarExpandIcon.js.map