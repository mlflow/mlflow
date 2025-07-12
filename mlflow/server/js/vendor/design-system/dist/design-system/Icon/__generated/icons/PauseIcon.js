import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPauseIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10 12V4h1.5v8zm-5.5 0V4H6v8z", clipRule: "evenodd" }) }));
}
const PauseIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPauseIcon });
});
PauseIcon.displayName = 'PauseIcon';
export default PauseIcon;
//# sourceMappingURL=PauseIcon.js.map