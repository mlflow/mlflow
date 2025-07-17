import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChevronDoubleDownIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M10.947 7.954 8 10.891 5.056 7.954 3.997 9.016l4.004 3.993 4.005-3.993z" }), _jsx("path", { fill: "currentColor", d: "M10.947 3.994 8 6.931 5.056 3.994 3.997 5.056 8.001 9.05l4.005-3.993z" })] }));
}
const ChevronDoubleDownIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChevronDoubleDownIcon });
});
ChevronDoubleDownIcon.displayName = 'ChevronDoubleDownIcon';
export default ChevronDoubleDownIcon;
//# sourceMappingURL=ChevronDoubleDownIcon.js.map