import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgH2Icon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.75 3A2.75 2.75 0 0 0 9 5.75V6h1.5v-.25c0-.69.56-1.25 1.25-1.25h.39a1.36 1.36 0 0 1 .746 2.498L10.692 8.44A3.75 3.75 0 0 0 9 11.574V13h6v-1.5h-4.499a2.25 2.25 0 0 1 1.014-1.807l2.194-1.44A2.86 2.86 0 0 0 12.14 3z" }) }));
}
const H2Icon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgH2Icon });
});
H2Icon.displayName = 'H2Icon';
export default H2Icon;
//# sourceMappingURL=H2Icon.js.map