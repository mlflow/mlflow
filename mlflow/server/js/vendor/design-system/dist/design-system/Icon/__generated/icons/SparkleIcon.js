import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSparkleIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10.726 8.813 13.199 8l-2.473-.813a3 3 0 0 1-1.913-1.913L8 2.801l-.813 2.473a3 3 0 0 1-1.913 1.913L2.801 8l2.473.813a3 3 0 0 1 1.913 1.913L8 13.199l.813-2.473a3 3 0 0 1 1.913-1.913m2.941.612c1.376-.452 1.376-2.398 0-2.85l-2.472-.813a1.5 1.5 0 0 1-.957-.956l-.813-2.473c-.452-1.376-2.398-1.376-2.85 0l-.813 2.473a1.5 1.5 0 0 1-.956.956l-2.473.813c-1.376.452-1.376 2.398 0 2.85l2.473.813a1.5 1.5 0 0 1 .956.957l.813 2.472c.452 1.376 2.398 1.376 2.85 0l.813-2.472a1.5 1.5 0 0 1 .957-.957z", clipRule: "evenodd" }) }));
}
const SparkleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSparkleIcon });
});
SparkleIcon.displayName = 'SparkleIcon';
export default SparkleIcon;
//# sourceMappingURL=SparkleIcon.js.map