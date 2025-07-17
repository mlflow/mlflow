import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgExpandMoreIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 17", ...props, children: _jsx("path", { fill: "currentColor", d: "m4 4.03 1.06 1.061 2.97-2.97L11 5.091l1.06-1.06L8.03 0zM12.06 12.091l-4.03 4.03L4 12.091l1.06-1.06L8.03 14 11 11.03z" }) }));
}
const ExpandMoreIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgExpandMoreIcon });
});
ExpandMoreIcon.displayName = 'ExpandMoreIcon';
export default ExpandMoreIcon;
//# sourceMappingURL=ExpandMoreIcon.js.map