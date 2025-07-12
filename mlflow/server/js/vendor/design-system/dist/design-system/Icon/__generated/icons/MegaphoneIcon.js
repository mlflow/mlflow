import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMegaphoneIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M12.197 1.243A.75.75 0 0 1 12.75 1h1.5a.75.75 0 0 1 .75.75v10.5a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1-.553-.243l-.892-.973A5.5 5.5 0 0 0 8 10.051V13a2 2 0 1 1-4 0v-3a3 3 0 0 1 0-6h3.25a5.5 5.5 0 0 0 4.055-1.784zM6.5 8.5v-3H4a1.5 1.5 0 1 0 0 3zm-1 1.5v3a.5.5 0 0 0 1 0v-3zm6.911.77A7 7 0 0 0 8 8.54V5.46a7 7 0 0 0 4.411-2.23l.669-.73h.42v9h-.42z", clipRule: "evenodd" }) }));
}
const MegaphoneIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMegaphoneIcon });
});
MegaphoneIcon.displayName = 'MegaphoneIcon';
export default MegaphoneIcon;
//# sourceMappingURL=MegaphoneIcon.js.map