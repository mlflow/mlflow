import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronDoubleLeftIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M8.047 10.944 5.11 8l2.937-2.944-1.062-1.06L2.991 8l3.994 4.003z" }), _jsx("path", { fill: "currentColor", d: "M12.008 10.944 9.07 8l2.938-2.944-1.062-1.06L6.952 8l3.994 4.003z" })] }));
}
const ChevronDoubleLeftIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronDoubleLeftIcon });
});
ChevronDoubleLeftIcon.displayName = 'ChevronDoubleLeftIcon';
export default ChevronDoubleLeftIcon;
//# sourceMappingURL=ChevronDoubleLeftIcon.js.map