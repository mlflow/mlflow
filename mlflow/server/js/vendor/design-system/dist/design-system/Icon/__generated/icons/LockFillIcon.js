import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLockFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M12 6V4a4 4 0 0 0-8 0v2H2.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75zM5.5 6h5V4a2.5 2.5 0 0 0-5 0zm1.75 7V9h1.5v4z", clipRule: "evenodd" }) }));
}
const LockFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLockFillIcon });
});
LockFillIcon.displayName = 'LockFillIcon';
export default LockFillIcon;
//# sourceMappingURL=LockFillIcon.js.map