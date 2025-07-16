import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBlockQuoteIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M16 2H0v1.5h16zM16 5.5H8V7h8zM16 9H8v1.5h8zM0 12.5V14h16v-1.5zM1.5 7.25A.25.25 0 0 1 1.75 7h.75V5.5h-.75A1.75 1.75 0 0 0 0 7.25v2.5c0 .414.336.75.75.75h1.5A.75.75 0 0 0 3 9.75v-1.5a.75.75 0 0 0-.75-.75H1.5zM5.5 7.5h.75a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-.75.75h-1.5A.75.75 0 0 1 4 9.75v-2.5c0-.966.784-1.75 1.75-1.75h.75V7h-.75a.25.25 0 0 0-.25.25z" }) }));
}
const BlockQuoteIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBlockQuoteIcon });
});
BlockQuoteIcon.displayName = 'BlockQuoteIcon';
export default BlockQuoteIcon;
//# sourceMappingURL=BlockQuoteIcon.js.map