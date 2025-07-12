import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBranchIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 4a3 3 0 1 1 5.186 2.055 3.23 3.23 0 0 0 2 1.155 3.001 3.001 0 1 1-.152 1.494A4.73 4.73 0 0 1 4.911 6.86a3 3 0 0 1-.161.046v2.19a3.001 3.001 0 1 1-1.5 0v-2.19A3 3 0 0 1 1 4m3-1.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3M2.5 12a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0m7-3.75a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0", clipRule: "evenodd" }) }));
}
const BranchIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBranchIcon });
});
BranchIcon.displayName = 'BranchIcon';
export default BranchIcon;
//# sourceMappingURL=BranchIcon.js.map