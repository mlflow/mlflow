import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M6.42 5.415A.75.75 0 0 0 5 5.75V11h1.5V8.927l.83 1.658a.75.75 0 0 0 1.34 0l.83-1.658V11H11V5.75a.75.75 0 0 0-1.42-.335L8 8.573z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z", clipRule: "evenodd" })] }));
}
const MIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMIcon });
});
MIcon.displayName = 'MIcon';
export default MIcon;
//# sourceMappingURL=MIcon.js.map