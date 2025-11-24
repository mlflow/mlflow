import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAlignRightIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 2.5h14V1H1zM15 5.75H8v-1.5h7zM1 8.75v-1.5h14v1.5zM1 15v-1.5h14V15zM8 11.75h7v-1.5H8z" }) }));
}
const AlignRightIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAlignRightIcon });
});
AlignRightIcon.displayName = 'AlignRightIcon';
export default AlignRightIcon;
//# sourceMappingURL=AlignRightIcon.js.map