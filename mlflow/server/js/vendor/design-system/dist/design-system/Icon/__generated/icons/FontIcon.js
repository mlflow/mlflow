import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFontIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { clipPath: "url(#FontIcon_svg__a)", children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.197 3.473a.75.75 0 0 0-1.393-.001L-.006 13H1.61l.6-1.5h4.562l.596 1.5h1.614zM6.176 10 4.498 5.776 2.809 10zm4.07-2.385c.593-.205 1.173-.365 1.754-.365a1.5 1.5 0 0 1 1.42 1.014A3.8 3.8 0 0 0 12 8c-.741 0-1.47.191-2.035.607A2.3 2.3 0 0 0 9 10.5c0 .81.381 1.464.965 1.893.565.416 1.294.607 2.035.607.524 0 1.042-.096 1.5-.298V13H15V8.75a3 3 0 0 0-3-3c-.84 0-1.614.23-2.245.448zM13.5 10.5a.8.8 0 0 0-.353-.685C12.897 9.631 12.5 9.5 12 9.5c-.5 0-.897.131-1.146.315a.8.8 0 0 0-.354.685c0 .295.123.515.354.685.25.184.645.315 1.146.315.502 0 .897-.131 1.147-.315.23-.17.353-.39.353-.685", clipRule: "evenodd" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const FontIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFontIcon });
});
FontIcon.displayName = 'FontIcon';
export default FontIcon;
//# sourceMappingURL=FontIcon.js.map