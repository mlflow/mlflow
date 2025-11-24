import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBriefcaseIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 4H5V2.75C5 1.784 5.784 1 6.75 1h2.5c.966 0 1.75.784 1.75 1.75V4h3.25a.75.75 0 0 1 .75.75v9.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-9.5A.75.75 0 0 1 1.75 4m5-1.5a.25.25 0 0 0-.25.25V4h3V2.75a.25.25 0 0 0-.25-.25zM2.5 8.173V13.5h11V8.173A9.2 9.2 0 0 1 8 9.985a9.2 9.2 0 0 1-5.5-1.812m0-1.978A7.72 7.72 0 0 0 8 8.485c2.15 0 4.095-.875 5.5-2.29V5.5h-11z", clipRule: "evenodd" }) }));
}
const BriefcaseIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBriefcaseIcon });
});
BriefcaseIcon.displayName = 'BriefcaseIcon';
export default BriefcaseIcon;
//# sourceMappingURL=BriefcaseIcon.js.map