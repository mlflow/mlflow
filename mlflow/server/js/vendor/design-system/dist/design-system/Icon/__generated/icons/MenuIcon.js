import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMenuIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M15 4H1V2.5h14zm0 4.75H1v-1.5h14zm0 4.75H1V12h14z", clipRule: "evenodd" }) }));
}
const MenuIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMenuIcon });
});
MenuIcon.displayName = 'MenuIcon';
export default MenuIcon;
//# sourceMappingURL=MenuIcon.js.map