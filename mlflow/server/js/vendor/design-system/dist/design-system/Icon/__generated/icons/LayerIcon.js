import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLayerIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M13.5 2.5H7V1h7.25a.75.75 0 0 1 .75.75V9h-1.5z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 7.75A.75.75 0 0 1 1.75 7h6.5a.75.75 0 0 1 .75.75v6.5a.75.75 0 0 1-.75.75h-6.5a.75.75 0 0 1-.75-.75zm1.5.75v5h5v-5z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M4 5.32h6.5V12H12V4.57a.75.75 0 0 0-.75-.75H4z" })] }));
}
const LayerIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLayerIcon });
});
LayerIcon.displayName = 'LayerIcon';
export default LayerIcon;
//# sourceMappingURL=LayerIcon.js.map