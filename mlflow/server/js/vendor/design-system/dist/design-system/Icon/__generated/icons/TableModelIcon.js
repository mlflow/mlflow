import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTableModelIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H6.5V7H15V1.75a.75.75 0 0 0-.75-.75zM5 7v6.5H2.5V7zm8.5-1.5v-3h-11v3z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.25 8.5a1.25 1.25 0 1 1 2.488.177l1.48 1.481a2 2 0 0 1 1.563 0l.731-.731a1.25 1.25 0 1 1 1.06 1.06l-.73.732a2 2 0 0 1 0 1.562l.731.731a1.25 1.25 0 1 1-1.06 1.06l-.732-.73a2 2 0 0 1-2.636-1.092H9.5a1.25 1.25 0 1 1 0-1.5h.645l.013-.031-1.481-1.481A1.25 1.25 0 0 1 7.25 8.5M11.5 12a.5.5 0 1 1 1 0 .5.5 0 0 1-1 0", clipRule: "evenodd" })] }));
}
const TableModelIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTableModelIcon });
});
TableModelIcon.displayName = 'TableModelIcon';
export default TableModelIcon;
//# sourceMappingURL=TableModelIcon.js.map