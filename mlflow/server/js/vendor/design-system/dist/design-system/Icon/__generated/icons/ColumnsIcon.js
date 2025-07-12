import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgColumnsIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zM2.5 13.5v-11H5v11zm4 0h3v-11h-3zm4.5-11v11h2.5v-11z", clipRule: "evenodd" }) }));
}
const ColumnsIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgColumnsIcon });
});
ColumnsIcon.displayName = 'ColumnsIcon';
export default ColumnsIcon;
//# sourceMappingURL=ColumnsIcon.js.map