import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCircleOutlineIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5 8a3 3 0 1 0 6 0 3 3 0 0 0-6 0m3-4.5a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9", clipRule: "evenodd" }) }));
}
const CircleOutlineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCircleOutlineIcon });
});
CircleOutlineIcon.displayName = 'CircleOutlineIcon';
export default CircleOutlineIcon;
//# sourceMappingURL=CircleOutlineIcon.js.map