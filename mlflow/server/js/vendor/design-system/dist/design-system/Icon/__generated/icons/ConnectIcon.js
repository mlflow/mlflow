import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgConnectIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.78 3.97 5.03 1.22a.75.75 0 0 0-1.06 0L1.22 3.97a.75.75 0 0 0 0 1.06l2.75 2.75a.75.75 0 0 0 1.06 0l2.75-2.75a.75.75 0 0 0 0-1.06m-1.59.53L4.5 6.19 2.81 4.5 4.5 2.81zM15 11.75a3.25 3.25 0 1 0-6.5 0 3.25 3.25 0 0 0 6.5 0M11.75 10a1.75 1.75 0 1 1 0 3.5 1.75 1.75 0 0 1 0-3.5", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M14.25 1H9v1.5h4.5V7H15V1.75a.75.75 0 0 0-.75-.75M1 9v5.25c0 .414.336.75.75.75H7v-1.5H2.5V9z" })] }));
}
const ConnectIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgConnectIcon });
});
ConnectIcon.displayName = 'ConnectIcon';
export default ConnectIcon;
//# sourceMappingURL=ConnectIcon.js.map