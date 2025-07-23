import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCalendarIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.5 0v2H1.75a.75.75 0 0 0-.75.75v11.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75H11.5V0H10v2H6V0zm9 3.5v2h-11v-2zM2.5 7v6.5h11V7z", clipRule: "evenodd" }) }));
}
const CalendarIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCalendarIcon });
});
CalendarIcon.displayName = 'CalendarIcon';
export default CalendarIcon;
//# sourceMappingURL=CalendarIcon.js.map