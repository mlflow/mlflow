import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgColumnIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M6.5 9V6h3v3zm3 1.5v3h-3v-3zm1.5-.75v-9a.75.75 0 0 0-.75-.75h-4.5A.75.75 0 0 0 5 .75v13.5c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75zM6.5 4.5v-3h3v3z", clipRule: "evenodd" }) }));
}
const ColumnIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgColumnIcon });
});
ColumnIcon.displayName = 'ColumnIcon';
export default ColumnIcon;
//# sourceMappingURL=ColumnIcon.js.map