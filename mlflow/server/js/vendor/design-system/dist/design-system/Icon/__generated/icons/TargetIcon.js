import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTargetIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M1.75 1H6v1.5H2.5V6H1V1.75A.75.75 0 0 1 1.75 1M14.25 1H10v1.5h3.5V6H15V1.75a.75.75 0 0 0-.75-.75M10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10zM2.5 13.5V10H1v4.25c0 .414.336.75.75.75H6v-1.5z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6.5 8a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0", clipRule: "evenodd" })] }));
}
const TargetIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTargetIcon });
});
TargetIcon.displayName = 'TargetIcon';
export default TargetIcon;
//# sourceMappingURL=TargetIcon.js.map