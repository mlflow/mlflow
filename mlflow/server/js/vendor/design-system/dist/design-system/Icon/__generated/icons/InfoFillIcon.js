import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgInfoFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0m-8.75 3V7h1.5v4zM8 4.5A.75.75 0 1 1 8 6a.75.75 0 0 1 0-1.5", clipRule: "evenodd" }) }));
}
const InfoFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgInfoFillIcon });
});
InfoFillIcon.displayName = 'InfoFillIcon';
export default InfoFillIcon;
//# sourceMappingURL=InfoFillIcon.js.map