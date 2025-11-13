import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFullscreenExitIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M6 1v4.25a.75.75 0 0 1-.75.75H1V4.5h3.5V1zM10 15v-4.25a.75.75 0 0 1 .75-.75H15v1.5h-3.5V15zM10.75 6H15V4.5h-3.5V1H10v4.25c0 .414.336.75.75.75M1 10h4.25a.75.75 0 0 1 .75.75V15H4.5v-3.5H1z" }) }));
}
const FullscreenExitIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFullscreenExitIcon });
});
FullscreenExitIcon.displayName = 'FullscreenExitIcon';
export default FullscreenExitIcon;
//# sourceMappingURL=FullscreenExitIcon.js.map