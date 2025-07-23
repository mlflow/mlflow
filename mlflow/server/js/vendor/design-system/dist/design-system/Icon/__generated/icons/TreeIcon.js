import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTreeIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2.004 9.602a2.751 2.751 0 1 0 3.371 3.47 2.751 2.751 0 0 0 5.25 0 2.751 2.751 0 1 0 3.371-3.47A2.75 2.75 0 0 0 11.25 7h-2.5v-.604a2.751 2.751 0 1 0-1.5 0V7h-2.5a2.75 2.75 0 0 0-2.746 2.602M2.75 11a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m4.5-2.5h-2.5a1.25 1.25 0 0 0-1.242 1.106 2.76 2.76 0 0 1 1.867 1.822A2.76 2.76 0 0 1 7.25 9.604zm1.5 0v1.104c.892.252 1.6.942 1.875 1.824a2.76 2.76 0 0 1 1.867-1.822A1.25 1.25 0 0 0 11.25 8.5zM12 12.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0m-5.25 0a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0M8 5a1.25 1.25 0 1 1 0-2.5A1.25 1.25 0 0 1 8 5", clipRule: "evenodd" }) }));
}
const TreeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTreeIcon });
});
TreeIcon.displayName = 'TreeIcon';
export default TreeIcon;
//# sourceMappingURL=TreeIcon.js.map