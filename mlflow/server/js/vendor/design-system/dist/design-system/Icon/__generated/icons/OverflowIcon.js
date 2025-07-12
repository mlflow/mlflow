import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgOverflowIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M8 1a1.75 1.75 0 1 0 0 3.5A1.75 1.75 0 0 0 8 1M8 6.25a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M8 11.5A1.75 1.75 0 1 0 8 15a1.75 1.75 0 0 0 0-3.5" }) }));
}
const OverflowIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgOverflowIcon });
});
OverflowIcon.displayName = 'OverflowIcon';
export default OverflowIcon;
//# sourceMappingURL=OverflowIcon.js.map