import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCalendarClockIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75H6v-1.5H2.5V7H15V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 5.5v-2h-11v2z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M10.25 10.5V12c0 .199.079.39.22.53l1 1 1.06-1.06-.78-.78V10.5z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7 12a4 4 0 1 1 8 0 4 4 0 0 1-8 0m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5", clipRule: "evenodd" })] }));
}
const CalendarClockIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCalendarClockIcon });
});
CalendarClockIcon.displayName = 'CalendarClockIcon';
export default CalendarClockIcon;
//# sourceMappingURL=CalendarClockIcon.js.map