import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgKeyIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 8a4 4 0 0 1 7.93-.75h7.32A.75.75 0 0 1 16 8v3h-1.5V8.75H13V11h-1.5V8.75H7.93A4.001 4.001 0 0 1 0 8m4-2.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5", clipRule: "evenodd" }) }));
}
const KeyIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgKeyIcon });
});
KeyIcon.displayName = 'KeyIcon';
export default KeyIcon;
//# sourceMappingURL=KeyIcon.js.map