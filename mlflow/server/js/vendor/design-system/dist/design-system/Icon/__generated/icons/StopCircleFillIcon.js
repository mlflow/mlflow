import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgStopCircleFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16M6.125 5.5a.625.625 0 0 0-.625.625v3.75c0 .345.28.625.625.625h3.75c.345 0 .625-.28.625-.625v-3.75a.625.625 0 0 0-.625-.625z", clipRule: "evenodd" }) }));
}
const StopCircleFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgStopCircleFillIcon });
});
StopCircleFillIcon.displayName = 'StopCircleFillIcon';
export default StopCircleFillIcon;
//# sourceMappingURL=StopCircleFillIcon.js.map