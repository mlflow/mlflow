import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarChartIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M1 1v13.25c0 .414.336.75.75.75H15v-1.5H2.5V1z" }), _jsx("path", { fill: "currentColor", d: "M7 1v11h1.5V1zM10 5v7h1.5V5zM4 5v7h1.5V5zM13 12V8h1.5v4z" })] }));
}
const BarChartIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarChartIcon });
});
BarChartIcon.displayName = 'BarChartIcon';
export default BarChartIcon;
//# sourceMappingURL=BarChartIcon.js.map