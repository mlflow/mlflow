import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgShortcutIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M14.25 14H9v-1.5h4.5v-10h-10V6H2V1.75A.75.75 0 0 1 2.75 1h11.5a.75.75 0 0 1 .75.75v11.5a.75.75 0 0 1-.75.75" }), _jsx("path", { fill: "currentColor", d: "M2 8h5v5H5.5v-2.872a2.251 2.251 0 0 0 .75 4.372V16A3.75 3.75 0 0 1 3.7 9.5H2z" })] }));
}
const ShortcutIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgShortcutIcon });
});
ShortcutIcon.displayName = 'ShortcutIcon';
export default ShortcutIcon;
//# sourceMappingURL=ShortcutIcon.js.map