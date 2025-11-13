import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgHomeIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.625 1.1a.75.75 0 0 1 .75 0l6.25 3.61a.75.75 0 0 1 .375.65v8.89a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V10H7v4.25a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75V5.355a.75.75 0 0 1 .375-.65zM2.5 5.79V13.5h3V9.25a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v4.25h3V5.792L8 2.616z", clipRule: "evenodd" }) }));
}
const HomeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgHomeIcon });
});
HomeIcon.displayName = 'HomeIcon';
export default HomeIcon;
//# sourceMappingURL=HomeIcon.js.map