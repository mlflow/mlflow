import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSearchIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { clipPath: "url(#SearchIcon_svg__a)", children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 1a7 7 0 1 0 4.39 12.453l2.55 2.55 1.06-1.06-2.55-2.55A7 7 0 0 0 8 1M2.5 8a5.5 5.5 0 1 1 11 0 5.5 5.5 0 0 1-11 0", clipRule: "evenodd" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 0h16v16H0z" }) }) })] }));
}
const SearchIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSearchIcon });
});
SearchIcon.displayName = 'SearchIcon';
export default SearchIcon;
//# sourceMappingURL=SearchIcon.js.map