import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAzHorizontalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.346 4.5a.75.75 0 0 0-.695.468L1 11.5h1.619l.406-1h2.643l.406 1h1.619L5.04 4.968a.75.75 0 0 0-.695-.468M5.06 9H3.634l.712-1.756zM12.667 6H9V4.5h5.25a.75.75 0 0 1 .58 1.225L11.333 10H15v1.5H9.75a.75.75 0 0 1-.58-1.225z", clipRule: "evenodd" }) }));
}
const AzHorizontalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAzHorizontalIcon });
});
AzHorizontalIcon.displayName = 'AzHorizontalIcon';
export default AzHorizontalIcon;
//# sourceMappingURL=AzHorizontalIcon.js.map