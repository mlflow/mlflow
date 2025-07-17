import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronDoubleUpIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M5.056 8.047 8 5.11l2.944 2.937 1.06-1.062L8 2.991 3.997 6.985z" }), _jsx("path", { fill: "currentColor", d: "M5.056 12.008 8 9.07l2.944 2.937 1.06-1.062L8 6.952l-4.003 3.994z" })] }));
}
const ChevronDoubleUpIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronDoubleUpIcon });
});
ChevronDoubleUpIcon.displayName = 'ChevronDoubleUpIcon';
export default ChevronDoubleUpIcon;
//# sourceMappingURL=ChevronDoubleUpIcon.js.map