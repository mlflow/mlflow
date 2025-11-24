import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgListIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1.5 2.75a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0M3 2h13v1.5H3zM3 5.5h13V7H3zM3 9h13v1.5H3zM3 12.5h13V14H3zM.75 7a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5M1.5 13.25a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0M.75 10.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5" }) }));
}
const ListIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgListIcon });
});
ListIcon.displayName = 'ListIcon';
export default ListIcon;
//# sourceMappingURL=ListIcon.js.map