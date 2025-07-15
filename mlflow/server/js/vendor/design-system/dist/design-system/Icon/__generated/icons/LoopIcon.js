import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLoopIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 17", ...props, children: _jsx("path", { fill: "currentColor", d: "M3.75 2A2.75 2.75 0 0 0 1 4.75v6.5A2.75 2.75 0 0 0 3.75 14H5.5v-1.5H3.75c-.69 0-1.25-.56-1.25-1.25v-6.5c0-.69.56-1.25 1.25-1.25h8.5c.69 0 1.25.56 1.25 1.25v6.5c0 .69-.56 1.25-1.25 1.25H9.81l.97-.97-1.06-1.06-2.78 2.78 2.78 2.78 1.06-1.06-.97-.97h2.44A2.75 2.75 0 0 0 15 11.25v-6.5A2.75 2.75 0 0 0 12.25 2z" }) }));
}
const LoopIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLoopIcon });
});
LoopIcon.displayName = 'LoopIcon';
export default LoopIcon;
//# sourceMappingURL=LoopIcon.js.map