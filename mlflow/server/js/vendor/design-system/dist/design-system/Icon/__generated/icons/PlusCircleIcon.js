import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlusCircleIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 11.5V8.75H4.5v-1.5h2.75V4.5h1.5v2.75h2.75v1.5H8.75v2.75z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0", clipRule: "evenodd" })] }));
}
const PlusCircleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlusCircleIcon });
});
PlusCircleIcon.displayName = 'PlusCircleIcon';
export default PlusCircleIcon;
//# sourceMappingURL=PlusCircleIcon.js.map