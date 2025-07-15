import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgForkIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 2.75a2.75 2.75 0 1 1 3.5 2.646V6.75h3.75A2.75 2.75 0 0 1 12 9.5v.104a2.751 2.751 0 1 1-1.5 0V9.5c0-.69-.56-1.25-1.25-1.25H5.5v1.354a2.751 2.751 0 1 1-1.5 0V5.396A2.75 2.75 0 0 1 2 2.75M4.75 1.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M3.5 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m6.5 0a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0", clipRule: "evenodd" }) }));
}
const ForkIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgForkIcon });
});
ForkIcon.displayName = 'ForkIcon';
export default ForkIcon;
//# sourceMappingURL=ForkIcon.js.map