import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgListBorderIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M12 8.75H7v-1.5h5zM7 5.5h5V4H7zM12 12H7v-1.5h5zM4.75 5.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M5.5 8A.75.75 0 1 1 4 8a.75.75 0 0 1 1.5 0M4.75 12a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v11h11v-11z", clipRule: "evenodd" })] }));
}
const ListBorderIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgListBorderIcon });
});
ListBorderIcon.displayName = 'ListBorderIcon';
export default ListBorderIcon;
//# sourceMappingURL=ListBorderIcon.js.map