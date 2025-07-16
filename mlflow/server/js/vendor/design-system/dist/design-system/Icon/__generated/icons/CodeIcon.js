import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCodeIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 17 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M4.03 12.06 5.091 11l-2.97-2.97 2.97-2.97L4.031 4 0 8.03zM12.091 4l4.03 4.03-4.03 4.03-1.06-1.06L14 8.03l-2.97-2.97z" }) }));
}
const CodeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCodeIcon });
});
CodeIcon.displayName = 'CodeIcon';
export default CodeIcon;
//# sourceMappingURL=CodeIcon.js.map