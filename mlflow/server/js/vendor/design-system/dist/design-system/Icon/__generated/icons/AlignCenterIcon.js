import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAlignCenterIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 2.5h14V1H1zM11.5 5.75h-7v-1.5h7zM15 8.75H1v-1.5h14zM15 15H1v-1.5h14zM4.5 11.75h7v-1.5h-7z" }) }));
}
const AlignCenterIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAlignCenterIcon });
});
AlignCenterIcon.displayName = 'AlignCenterIcon';
export default AlignCenterIcon;
//# sourceMappingURL=AlignCenterIcon.js.map