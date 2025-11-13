import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCalendarRangeIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsxs("g", { fill: "currentColor", clipPath: "url(#CalendarRangeIcon_svg__a)", children: [_jsx("path", { fillRule: "evenodd", d: "M6 2h4V0h1.5v2h2.75a.75.75 0 0 1 .75.75V8.5h-1.5V7h-11v6.5H8V15H1.75a.75.75 0 0 1-.75-.75V2.75A.75.75 0 0 1 1.75 2H4.5V0H6zM2.5 5.5h11v-2h-11z", clipRule: "evenodd" }), _jsx("path", { d: "M10.47 9.47 7.94 12l2.53 2.53 1.06-1.06-.72-.72h2.38l-.72.72 1.06 1.06L16.06 12l-2.53-2.53-1.06 1.06.72.72h-2.38l.72-.72z" })] }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const CalendarRangeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCalendarRangeIcon });
});
CalendarRangeIcon.displayName = 'CalendarRangeIcon';
export default CalendarRangeIcon;
//# sourceMappingURL=CalendarRangeIcon.js.map