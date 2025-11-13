import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPageFirstIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m12.97 1 1.06 1.06-5.97 5.97L14.03 14l-1.06 1.06-7.03-7.03zM2.5 15.03H1v-14h1.5z", clipRule: "evenodd" }) }));
}
const PageFirstIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPageFirstIcon });
});
PageFirstIcon.displayName = 'PageFirstIcon';
export default PageFirstIcon;
//# sourceMappingURL=PageFirstIcon.js.map