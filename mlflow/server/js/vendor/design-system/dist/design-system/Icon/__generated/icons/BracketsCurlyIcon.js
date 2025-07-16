import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBracketsCurlyIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M5.5 2a2.75 2.75 0 0 0-2.75 2.75v1C2.75 6.44 2.19 7 1.5 7H1v2h.5c.69 0 1.25.56 1.25 1.25v1A2.75 2.75 0 0 0 5.5 14H6v-1.5h-.5c-.69 0-1.25-.56-1.25-1.25v-1c0-.93-.462-1.752-1.168-2.25A2.75 2.75 0 0 0 4.25 5.75v-1c0-.69.56-1.25 1.25-1.25H6V2zM13.25 4.75A2.75 2.75 0 0 0 10.5 2H10v1.5h.5c.69 0 1.25.56 1.25 1.25v1c0 .93.462 1.752 1.168 2.25a2.75 2.75 0 0 0-1.168 2.25v1c0 .69-.56 1.25-1.25 1.25H10V14h.5a2.75 2.75 0 0 0 2.75-2.75v-1c0-.69.56-1.25 1.25-1.25h.5V7h-.5c-.69 0-1.25-.56-1.25-1.25z" }) }));
}
const BracketsCurlyIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBracketsCurlyIcon });
});
BracketsCurlyIcon.displayName = 'BracketsCurlyIcon';
export default BracketsCurlyIcon;
//# sourceMappingURL=BracketsCurlyIcon.js.map