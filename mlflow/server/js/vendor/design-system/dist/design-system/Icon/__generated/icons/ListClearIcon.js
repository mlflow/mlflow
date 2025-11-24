import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgListClearIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("g", { fill: "currentColor", clipPath: "url(#ListClearIcon_svg__a)", children: _jsx("path", { d: "M15.03 13.97 13.06 12l1.97-1.97-1.06-1.06L12 10.94l-1.97-1.97-1.06 1.06L10.94 12l-1.97 1.97 1.06 1.06L12 13.06l1.97 1.97zM5 11.5H1V10h4zM11 3.5H1V2h10zM7 7.5H1V6h6z" }) }), _jsx("defs", { children: _jsx("clipPath", { children: _jsx("path", { fill: "#fff", d: "M0 16h16V0H0z" }) }) })] }));
}
const ListClearIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgListClearIcon });
});
ListClearIcon.displayName = 'ListClearIcon';
export default ListClearIcon;
//# sourceMappingURL=ListClearIcon.js.map