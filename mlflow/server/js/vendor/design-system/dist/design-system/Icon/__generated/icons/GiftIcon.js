import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgGiftIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3 3.25A2.25 2.25 0 0 1 5.25 1C6.365 1 7.36 1.522 8 2.335A3.5 3.5 0 0 1 10.75 1a2.25 2.25 0 0 1 2.122 3h1.378a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75H14v5.75a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75V8.5h-.25A.75.75 0 0 1 1 7.75v-3A.75.75 0 0 1 1.75 4h1.378A2.3 2.3 0 0 1 3 3.25M5.25 4h1.937A2 2 0 0 0 5.25 2.5a.75.75 0 0 0 0 1.5m2 1.5H2.5V7h4.75zm0 3H3.5v5h3.75zm1.5 5v-5h3.75v5zm0-6.5V5.5h4.75V7zm.063-3h1.937a.75.75 0 0 0 0-1.5A2 2 0 0 0 8.813 4", clipRule: "evenodd" }) }));
}
const GiftIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgGiftIcon });
});
GiftIcon.displayName = 'GiftIcon';
export default GiftIcon;
//# sourceMappingURL=GiftIcon.js.map