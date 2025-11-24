import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPaperclipIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M11.536 2.343a2.25 2.25 0 0 0-3.182 0l-4.95 4.95a3.75 3.75 0 1 0 5.303 5.303l4.066-4.066 1.06 1.06-4.065 4.067a5.25 5.25 0 1 1-7.425-7.425l4.95-4.95a3.75 3.75 0 1 1 5.303 5.304l-4.95 4.95a2.25 2.25 0 1 1-3.182-3.182l5.48-5.48 1.061 1.06-5.48 5.48a.75.75 0 1 0 1.06 1.06l4.95-4.949a2.25 2.25 0 0 0 0-3.182", clipRule: "evenodd" }) }));
}
const PaperclipIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPaperclipIcon });
});
PaperclipIcon.displayName = 'PaperclipIcon';
export default PaperclipIcon;
//# sourceMappingURL=PaperclipIcon.js.map