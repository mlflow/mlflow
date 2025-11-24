import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgXCircleFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m1.97-4.97L8 9.06l-1.97 1.97-1.06-1.06L6.94 8 4.97 6.03l1.06-1.06L8 6.94l1.97-1.97 1.06 1.06L9.06 8l1.97 1.97z", clipRule: "evenodd" }) }));
}
const XCircleFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgXCircleFillIcon });
});
XCircleFillIcon.displayName = 'XCircleFillIcon';
export default XCircleFillIcon;
//# sourceMappingURL=XCircleFillIcon.js.map