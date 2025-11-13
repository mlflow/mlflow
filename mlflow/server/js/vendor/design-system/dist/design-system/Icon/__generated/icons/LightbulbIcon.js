import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLightbulbIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 0v2h1.5V0zM16 7.25h-2v1.5h2zM0 7.25h2v1.5H0zM13.127 1.813l-1.415 1.414 1.061 1.06 1.414-1.414zM2.874 1.813l1.414 1.414-1.06 1.06-1.415-1.414z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3.25 8.221C3.25 5.61 5.382 3.5 8 3.5s4.75 2.109 4.75 4.721a4.7 4.7 0 0 1-.985 2.879c-.754.973-1.33 1.776-1.33 2.644v1.506a.75.75 0 0 1-.75.75h-3.37a.75.75 0 0 1-.75-.75v-1.506c0-.868-.576-1.67-1.33-2.644A4.7 4.7 0 0 1 3.25 8.22M8 5C6.2 5 4.75 6.447 4.75 8.221c0 .738.25 1.417.67 1.96l.044.056c.284.366.612.789.897 1.263h3.278c.285-.474.613-.897.897-1.263l.043-.056c.422-.543.671-1.222.671-1.96C11.25 6.447 9.8 5 8 5m-.934 8.744c0-.256-.03-.504-.081-.744h2.03q-.079.36-.08.744v.756h-1.87z", clipRule: "evenodd" })] }));
}
const LightbulbIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLightbulbIcon });
});
LightbulbIcon.displayName = 'LightbulbIcon';
export default LightbulbIcon;
//# sourceMappingURL=LightbulbIcon.js.map