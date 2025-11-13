import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgClipboardIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.5 0a.75.75 0 0 0-.75.75V1h-2a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75h-2V.75A.75.75 0 0 0 10.5 0zm5.75 2.5v.75a.75.75 0 0 1-.75.75h-5a.75.75 0 0 1-.75-.75V2.5H3.5v11h9v-11zm-5 0v-1h3.5v1z", clipRule: "evenodd" }) }));
}
const ClipboardIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgClipboardIcon });
});
ClipboardIcon.displayName = 'ClipboardIcon';
export default ClipboardIcon;
//# sourceMappingURL=ClipboardIcon.js.map