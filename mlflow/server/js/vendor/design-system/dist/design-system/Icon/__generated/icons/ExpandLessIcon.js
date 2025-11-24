import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgExpandLessIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 17", ...props, children: _jsx("path", { fill: "currentColor", d: "M12.06 1.06 11 0 8.03 2.97 5.06 0 4 1.06l4.03 4.031zM4 15l4.03-4.03L12.06 15 11 16.06l-2.97-2.969-2.97 2.97z" }) }));
}
const ExpandLessIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgExpandLessIcon });
});
ExpandLessIcon.displayName = 'ExpandLessIcon';
export default ExpandLessIcon;
//# sourceMappingURL=ExpandLessIcon.js.map