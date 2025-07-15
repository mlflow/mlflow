import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgKeyboardIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75V2.75a.75.75 0 0 0-.75-.75zm.75 10.5v-9h13v9zm2.75-8h-1.5V6h1.5zm1.5 0V6h1.5V4.5zm3 0V6h1.5V4.5zm3 0V6h1.5V4.5zm-1.5 2.75h-1.5v1.5h1.5zm1.5 1.5v-1.5h1.5v1.5zm-4.5 0v-1.5h-1.5v1.5zm-3 0v-1.5h-1.5v1.5zM11 10H5v1.5h6z", clipRule: "evenodd" }) }));
}
const KeyboardIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgKeyboardIcon });
});
KeyboardIcon.displayName = 'KeyboardIcon';
export default KeyboardIcon;
//# sourceMappingURL=KeyboardIcon.js.map