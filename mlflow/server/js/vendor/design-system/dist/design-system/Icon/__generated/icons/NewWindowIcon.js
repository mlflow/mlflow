import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgNewWindowIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M10 1h5v5h-1.5V3.56L8.53 8.53 7.47 7.47l4.97-4.97H10z" }), _jsx("path", { fill: "currentColor", d: "M1 2.75A.75.75 0 0 1 1.75 2H8v1.5H2.5v10h10V8H14v6.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75z" })] }));
}
const NewWindowIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgNewWindowIcon });
});
NewWindowIcon.displayName = 'NewWindowIcon';
export default NewWindowIcon;
//# sourceMappingURL=NewWindowIcon.js.map