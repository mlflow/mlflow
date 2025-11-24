import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCellsSquareIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75v4.75h4.75V2.5zm6.25 0v4.75h4.75V2.5zm-1.5 6.25H2.5v4.75h4.75zm1.5 4.75V8.75h4.75v4.75z", clipRule: "evenodd" }) }));
}
const CellsSquareIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCellsSquareIcon });
});
CellsSquareIcon.displayName = 'CellsSquareIcon';
export default CellsSquareIcon;
//# sourceMappingURL=CellsSquareIcon.js.map