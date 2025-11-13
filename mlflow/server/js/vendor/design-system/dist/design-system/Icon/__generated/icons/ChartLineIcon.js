import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChartLineIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z" }), _jsx("path", { fill: "currentColor", d: "m15.03 5.03-1.06-1.06L9.5 8.44 7 5.94 3.47 9.47l1.06 1.06L7 8.06l2.5 2.5z" })] }));
}
const ChartLineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChartLineIcon });
});
ChartLineIcon.displayName = 'ChartLineIcon';
export default ChartLineIcon;
//# sourceMappingURL=ChartLineIcon.js.map