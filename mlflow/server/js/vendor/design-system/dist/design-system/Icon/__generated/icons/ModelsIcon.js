import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgModelsIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { clipPath: "url(#ModelsIcon_svg__a)", children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 4.75a2.75 2.75 0 0 1 5.145-1.353l4.372-.95a2.75 2.75 0 1 1 3.835 2.823l.282 2.257a2.75 2.75 0 1 1-2.517 4.46l-2.62 1.145.003.118a2.75 2.75 0 1 1-4.415-2.19L3.013 7.489A2.75 2.75 0 0 1 0 4.75M2.75 3.5a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5m2.715 1.688q.027-.164.033-.333l4.266-.928a2.75 2.75 0 0 0 2.102 1.546l.282 2.257c-.377.165-.71.412-.976.719zM4.828 6.55a2.8 2.8 0 0 1-.413.388l1.072 3.573q.13-.012.263-.012c.945 0 1.778.476 2.273 1.202l2.5-1.093a2.8 2.8 0 0 1 .012-.797zM12 10.25a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0M5.75 12a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5M11 2.75a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0", clipRule: "evenodd" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const ModelsIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgModelsIcon });
});
ModelsIcon.displayName = 'ModelsIcon';
export default ModelsIcon;
//# sourceMappingURL=ModelsIcon.js.map