import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTextBoxIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11zM5 6h2.25v5.5h1.5V6H11V4.5H5z", clipRule: "evenodd" }) }));
}
const TextBoxIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTextBoxIcon });
});
TextBoxIcon.displayName = 'TextBoxIcon';
export default TextBoxIcon;
//# sourceMappingURL=TextBoxIcon.js.map