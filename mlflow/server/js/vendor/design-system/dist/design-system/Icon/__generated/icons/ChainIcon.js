import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgChainIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "m6.144 12.331.972-.972 1.06 1.06-.971.973a3.625 3.625 0 1 1-5.127-5.127l2.121-2.121A3.625 3.625 0 0 1 10.32 8H8.766a2.125 2.125 0 0 0-3.507-.795l-2.121 2.12a2.125 2.125 0 0 0 3.005 3.006" }), _jsx("path", { fill: "currentColor", d: "m9.856 3.669-.972.972-1.06-1.06.971-.973a3.625 3.625 0 1 1 5.127 5.127l-2.121 2.121A3.625 3.625 0 0 1 5.68 8h1.552a2.125 2.125 0 0 0 3.507.795l2.121-2.12a2.125 2.125 0 0 0-3.005-3.006" })] }));
}
const ChainIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgChainIcon });
});
ChainIcon.displayName = 'ChainIcon';
export default ChainIcon;
//# sourceMappingURL=ChainIcon.js.map