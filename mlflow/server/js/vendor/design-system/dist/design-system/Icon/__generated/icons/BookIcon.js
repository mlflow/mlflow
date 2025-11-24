import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBookIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2.75 1a.75.75 0 0 0-.75.75v13.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zM7.5 2.5h-4v6.055l1.495-1.36a.75.75 0 0 1 1.01 0L7.5 8.555zm-4 8.082 2-1.818 2.246 2.041A.75.75 0 0 0 9 10.25V2.5h3.5v12h-9z", clipRule: "evenodd" }) }));
}
const BookIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBookIcon });
});
BookIcon.displayName = 'BookIcon';
export default BookIcon;
//# sourceMappingURL=BookIcon.js.map