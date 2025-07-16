import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFlagPointerIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3 2.5h5.439a.5.5 0 0 1 .39.188l4 5a.5.5 0 0 1 0 .624l-4 5a.5.5 0 0 1-.39.188H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5M1 3a2 2 0 0 1 2-2h5.439A2 2 0 0 1 10 1.75l4 5a2 2 0 0 1 0 2.5l-4 5a2 2 0 0 1-1.562.75H3a2 2 0 0 1-2-2zm6 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4", clipRule: "evenodd" }) }));
}
const FlagPointerIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFlagPointerIcon });
});
FlagPointerIcon.displayName = 'FlagPointerIcon';
export default FlagPointerIcon;
//# sourceMappingURL=FlagPointerIcon.js.map