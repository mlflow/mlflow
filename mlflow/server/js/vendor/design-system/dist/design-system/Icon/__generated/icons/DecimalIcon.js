import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDecimalIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3 10a3 3 0 1 0 6 0V6a3 3 0 0 0-6 0zm3 1.5A1.5 1.5 0 0 1 4.5 10V6a1.5 1.5 0 1 1 3 0v4A1.5 1.5 0 0 1 6 11.5M10 10a3 3 0 1 0 6 0V6a3 3 0 1 0-6 0zm3 1.5a1.5 1.5 0 0 1-1.5-1.5V6a1.5 1.5 0 0 1 3 0v4a1.5 1.5 0 0 1-1.5 1.5", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M1 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2" })] }));
}
const DecimalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDecimalIcon });
});
DecimalIcon.displayName = 'DecimalIcon';
export default DecimalIcon;
//# sourceMappingURL=DecimalIcon.js.map