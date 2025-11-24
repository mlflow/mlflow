import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCalendarEventIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M8.5 10.25a1.75 1.75 0 1 1 3.5 0 1.75 1.75 0 0 1-3.5 0" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10 2H6V0H4.5v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10zM2.5 3.5v2h11v-2zm0 10V7h11v6.5z", clipRule: "evenodd" })] }));
}
const CalendarEventIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCalendarEventIcon });
});
CalendarEventIcon.displayName = 'CalendarEventIcon';
export default CalendarEventIcon;
//# sourceMappingURL=CalendarEventIcon.js.map