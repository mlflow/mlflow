import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCursorIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { clipPath: "url(#CursorIcon_svg__a)", children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.22 1.22a.75.75 0 0 1 .802-.169l13.5 5.25a.75.75 0 0 1-.043 1.413L9.597 9.597l-1.883 5.882a.75.75 0 0 1-1.413.043l-5.25-13.5a.75.75 0 0 1 .169-.802m1.847 1.847 3.864 9.937 1.355-4.233a.75.75 0 0 1 .485-.485l4.233-1.355z", clipRule: "evenodd" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M16 0H0v16h16z" }) }) })] }));
}
const CursorIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCursorIcon });
});
CursorIcon.displayName = 'CursorIcon';
export default CursorIcon;
//# sourceMappingURL=CursorIcon.js.map