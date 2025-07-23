import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlayIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M12.125 8.864a.75.75 0 0 0 0-1.3l-6-3.464A.75.75 0 0 0 5 4.75v6.928a.75.75 0 0 0 1.125.65z" }) }));
}
const PlayIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlayIcon });
});
PlayIcon.displayName = 'PlayIcon';
export default PlayIcon;
//# sourceMappingURL=PlayIcon.js.map