import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTextUnderlineIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8.5 1H10v3.961a3.4 3.4 0 0 1 1.75-.461c.857 0 1.674.287 2.283.863.616.582.967 1.411.967 2.387s-.351 1.805-.967 2.387c-.61.576-1.426.863-2.283.863a3.4 3.4 0 0 1-1.75-.461V11H8.5zM10 7.75c0 .602.208 1.023.498 1.297.295.28.728.453 1.252.453s.957-.174 1.252-.453c.29-.274.498-.695.498-1.297s-.208-1.023-.498-1.297C12.708 6.173 12.275 6 11.75 6s-.957.174-1.252.453c-.29.274-.498.695-.498 1.297M4 5.25c-.582 0-1.16.16-1.755.365l-.49-1.417C2.385 3.979 3.159 3.75 4 3.75a3 3 0 0 1 3 3V11H5.5v-.298A3.7 3.7 0 0 1 4 11c-.741 0-1.47-.191-2.035-.607A2.3 2.3 0 0 1 1 8.5c0-.81.381-1.464.965-1.893C2.529 6.19 3.259 6 4 6c.494 0 .982.085 1.42.264A1.5 1.5 0 0 0 4 5.25m1.147 2.565c.23.17.353.39.353.685a.8.8 0 0 1-.353.685C4.897 9.369 4.5 9.5 4 9.5s-.897-.131-1.147-.315A.8.8 0 0 1 2.5 8.5c0-.295.123-.515.353-.685C3.103 7.631 3.5 7.5 4 7.5s.897.131 1.147.315", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M1 12.5h14V14H1z" })] }));
}
const TextUnderlineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTextUnderlineIcon });
});
TextUnderlineIcon.displayName = 'TextUnderlineIcon';
export default TextUnderlineIcon;
//# sourceMappingURL=TextUnderlineIcon.js.map