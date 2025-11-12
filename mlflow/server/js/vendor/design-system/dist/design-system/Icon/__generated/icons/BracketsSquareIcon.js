import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBracketsSquareIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1H5v1.5H2.5v11H5V15H1.75a.75.75 0 0 1-.75-.75zm12.5.75H11V1h3.25a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H11v-1.5h2.5z", clipRule: "evenodd" }) }));
}
const BracketsSquareIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBracketsSquareIcon });
});
BracketsSquareIcon.displayName = 'BracketsSquareIcon';
export default BracketsSquareIcon;
//# sourceMappingURL=BracketsSquareIcon.js.map