import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTableMeasureIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75H4v-1.5H2.5V7H5v2h1.5V7h3v2H11V7h2.5v2H15V1.75a.75.75 0 0 0-.75-.75zM13.5 5.5v-3h-11v3z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "M5 11v3.25c0 .414.336.75.75.75h9.5a.75.75 0 0 0 .75-.75V11h-1.5v2.5h-.875V12h-1.5v1.5h-.875V11h-1.5v2.5h-.875V12h-1.5v1.5H6.5V11z" })] }));
}
const TableMeasureIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTableMeasureIcon });
});
TableMeasureIcon.displayName = 'TableMeasureIcon';
export default TableMeasureIcon;
//# sourceMappingURL=TableMeasureIcon.js.map