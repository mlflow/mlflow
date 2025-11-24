import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMeasureIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10.22.72a.75.75 0 0 1 1.06 0l4 4a.75.75 0 0 1 0 1.06l-9.5 9.5a.75.75 0 0 1-1.06 0l-4-4a.75.75 0 0 1 0-1.06zm.53 1.59-8.44 8.44 2.94 2.94 1.314-1.315-1.47-1.47 1.061-1.06 1.47 1.47L8.939 10 7.47 8.53 8.53 7.47 10 8.94l1.314-1.315-1.47-1.47 1.061-1.06 1.47 1.47 1.314-1.315z", clipRule: "evenodd" }) }));
}
const MeasureIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMeasureIcon });
});
MeasureIcon.displayName = 'MeasureIcon';
export default MeasureIcon;
//# sourceMappingURL=MeasureIcon.js.map