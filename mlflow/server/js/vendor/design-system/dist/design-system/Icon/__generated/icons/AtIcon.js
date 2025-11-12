import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAtIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2.5 8a5.5 5.5 0 1 1 11 0l-.002 1.08a.973.973 0 0 1-1.946-.002V4.984h-1.5v.194A3.52 3.52 0 0 0 8 4.5C6.22 4.5 4.5 5.949 4.5 8s1.72 3.5 3.5 3.5c.917 0 1.817-.384 2.475-1.037a2.473 2.473 0 0 0 4.523-1.38L15 8a7 7 0 1 0-3.137 5.839l-.83-1.25A5.5 5.5 0 0 1 2.5 8M6 8c0-1.153.976-2 2-2s2 .847 2 2-.976 2-2 2-2-.847-2-2", clipRule: "evenodd" }) }));
}
const AtIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAtIcon });
});
AtIcon.displayName = 'AtIcon';
export default AtIcon;
//# sourceMappingURL=AtIcon.js.map