import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgH1Icon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 3v10h1.5V8.75H6V13h1.5V3H6v4.25H2.5V3zM11.25 3A2.25 2.25 0 0 1 9 5.25v1.5c.844 0 1.623-.279 2.25-.75v5.5H9V13h6v-1.5h-2.25V3z" }) }));
}
const H1Icon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgH1Icon });
});
H1Icon.displayName = 'H1Icon';
export default H1Icon;
//# sourceMappingURL=H1Icon.js.map