import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSendIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M16 8a.75.75 0 0 1-.435.68l-13.5 6.25a.75.75 0 0 1-1.02-.934L3.202 8 1.044 2.004a.75.75 0 0 1 1.021-.935l13.5 6.25A.75.75 0 0 1 16 8m-11.473.75-1.463 4.065L13.464 8l-10.4-4.815L4.527 7.25H8v1.5z", clipRule: "evenodd" }) }));
}
const SendIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSendIcon });
});
SendIcon.displayName = 'SendIcon';
export default SendIcon;
//# sourceMappingURL=SendIcon.js.map