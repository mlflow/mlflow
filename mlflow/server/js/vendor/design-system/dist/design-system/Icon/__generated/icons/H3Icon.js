import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgH3Icon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 3h1.5v4.25H6V3h1.5v10H6V8.75H2.5V13H1zM9 5.75A2.75 2.75 0 0 1 11.75 3h.375a2.875 2.875 0 0 1 1.937 5 2.875 2.875 0 0 1-1.937 5h-.375A2.75 2.75 0 0 1 9 10.25V10h1.5v.25c0 .69.56 1.25 1.25 1.25h.375a1.375 1.375 0 1 0 0-2.75H11v-1.5h1.125a1.375 1.375 0 1 0 0-2.75h-.375c-.69 0-1.25.56-1.25 1.25V6H9z" }) }));
}
const H3Icon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgH3Icon });
});
H3Icon.displayName = 'H3Icon';
export default H3Icon;
//# sourceMappingURL=H3Icon.js.map