import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCheckSmallIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M12.03 6.03 7 11.06 3.97 8.03l1.06-1.06L7 8.94l3.97-3.97z", clipRule: "evenodd" }) }));
}
const CheckSmallIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCheckSmallIcon });
});
CheckSmallIcon.displayName = 'CheckSmallIcon';
export default CheckSmallIcon;
//# sourceMappingURL=CheckSmallIcon.js.map