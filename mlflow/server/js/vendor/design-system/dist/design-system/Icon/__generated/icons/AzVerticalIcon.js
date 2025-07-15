import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAzVerticalIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { fill: "currentColor", fillRule: "evenodd", clipPath: "url(#AZVerticalIcon_svg__a)", clipRule: "evenodd", children: _jsx("path", { d: "M7.996 0a.75.75 0 0 1 .695.468L11.343 7h-1.62l-.405-1H6.675l-.406 1H4.65L7.301.468A.75.75 0 0 1 7.996 0m-.712 4.5h1.425l-.713-1.756zM8.664 9.5H4.996V8h5.25a.75.75 0 0 1 .58 1.225L7.33 13.5h3.667V15h-5.25a.75.75 0 0 1-.58-1.225z" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const AzVerticalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAzVerticalIcon });
});
AzVerticalIcon.displayName = 'AzVerticalIcon';
export default AzVerticalIcon;
//# sourceMappingURL=AzVerticalIcon.js.map