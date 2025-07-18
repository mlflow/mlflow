import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSearchDataIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.651 3.128a.75.75 0 0 0-1.302 0l-1 1.75A.75.75 0 0 0 6 6h2a.75.75 0 0 0 .651-1.122zM4.75 6.5a1.75 1.75 0 1 0 0 3.5 1.75 1.75 0 0 0 0-3.5M7.5 7.25a.75.75 0 0 1 .75-.75h2a.75.75 0 0 1 .75.75v2a.75.75 0 0 1-.75.75h-2a.75.75 0 0 1-.75-.75z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 7a7 7 0 1 1 12.45 4.392l2.55 2.55-1.06 1.061-2.55-2.55A7 7 0 0 1 0 7m7-5.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11", clipRule: "evenodd" })] }));
}
const SearchDataIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSearchDataIcon });
});
SearchDataIcon.displayName = 'SearchDataIcon';
export default SearchDataIcon;
//# sourceMappingURL=SearchDataIcon.js.map