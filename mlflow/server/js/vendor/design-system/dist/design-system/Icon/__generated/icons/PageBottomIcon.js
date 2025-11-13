import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPageBottomIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 3.06 2.06 2l5.97 5.97L14 2l1.06 1.06-7.03 7.031zm14.03 10.47v1.5h-14v-1.5z", clipRule: "evenodd" }) }));
}
const PageBottomIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPageBottomIcon });
});
PageBottomIcon.displayName = 'PageBottomIcon';
export default PageBottomIcon;
//# sourceMappingURL=PageBottomIcon.js.map