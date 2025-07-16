import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFullscreenIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M6 1H1.75a.75.75 0 0 0-.75.75V6h1.5V2.5H6zM10 2.5V1h4.25a.75.75 0 0 1 .75.75V6h-1.5V2.5zM10 13.5h3.5V10H15v4.25a.75.75 0 0 1-.75.75H10zM2.5 10v3.5H6V15H1.75a.75.75 0 0 1-.75-.75V10z" }) }));
}
const FullscreenIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFullscreenIcon });
});
FullscreenIcon.displayName = 'FullscreenIcon';
export default FullscreenIcon;
//# sourceMappingURL=FullscreenIcon.js.map