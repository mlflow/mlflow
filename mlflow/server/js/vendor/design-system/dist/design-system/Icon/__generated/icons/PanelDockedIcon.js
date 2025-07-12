import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPanelDockedIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H14.3a.7.7 0 0 0 .7-.7V1.75a.75.75 0 0 0-.75-.75zM8 13.5V8.7a.7.7 0 0 1 .7-.7h4.8V2.5h-11v11z", clipRule: "evenodd" }) }));
}
const PanelDockedIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPanelDockedIcon });
});
PanelDockedIcon.displayName = 'PanelDockedIcon';
export default PanelDockedIcon;
//# sourceMappingURL=PanelDockedIcon.js.map