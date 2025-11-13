import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlayCircleIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M11.25 8a.75.75 0 0 1-.375.65l-3.75 2.165A.75.75 0 0 1 6 10.165v-4.33a.75.75 0 0 1 1.125-.65l3.75 2.165a.75.75 0 0 1 .375.65" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.5 8a6.5 6.5 0 1 1 13 0 6.5 6.5 0 0 1-13 0", clipRule: "evenodd" })] }));
}
const PlayCircleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlayCircleIcon });
});
PlayCircleIcon.displayName = 'PlayCircleIcon';
export default PlayCircleIcon;
//# sourceMappingURL=PlayCircleIcon.js.map