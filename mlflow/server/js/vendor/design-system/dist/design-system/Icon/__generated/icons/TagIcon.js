import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTagIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M5 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.768 1.018a.75.75 0 0 0-.75.75v6.1c0 .199.079.39.22.53l6.884 6.885a.75.75 0 0 0 1.06 0l6.101-6.1a.75.75 0 0 0 0-1.061L8.4 1.237a.75.75 0 0 0-.53-.22zm6.884 12.674L2.518 7.557v-5.04h5.04l6.134 6.135z", clipRule: "evenodd" })] }));
}
const TagIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTagIcon });
});
TagIcon.displayName = 'TagIcon';
export default TagIcon;
//# sourceMappingURL=TagIcon.js.map