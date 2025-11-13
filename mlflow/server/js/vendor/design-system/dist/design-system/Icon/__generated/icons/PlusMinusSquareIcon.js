import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlusMinusSquareIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 4.25V6H5.5v1.5h1.75v1.75h1.5V7.5h1.75V6H8.75V4.25zM10.5 10.5h-5V12h5z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 1a.75.75 0 0 0-.75.75v12.5c0 .414.336.75.75.75h12.5a.75.75 0 0 0 .75-.75V1.75a.75.75 0 0 0-.75-.75zm.75 12.5v-11h11v11z", clipRule: "evenodd" })] }));
}
const PlusMinusSquareIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlusMinusSquareIcon });
});
PlusMinusSquareIcon.displayName = 'PlusMinusSquareIcon';
export default PlusMinusSquareIcon;
//# sourceMappingURL=PlusMinusSquareIcon.js.map