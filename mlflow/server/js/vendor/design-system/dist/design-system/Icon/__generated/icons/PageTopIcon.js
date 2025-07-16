import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPageTopIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m1 12.97 1.06 1.06 5.97-5.97L14 14.03l1.06-1.06-7.03-7.03zM15.03 2.5V1h-14v1.5z", clipRule: "evenodd" }) }));
}
const PageTopIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPageTopIcon });
});
PageTopIcon.displayName = 'PageTopIcon';
export default PageTopIcon;
//# sourceMappingURL=PageTopIcon.js.map