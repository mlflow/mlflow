import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSaveIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M10 9.25H6v1.5h4z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1H11a.75.75 0 0 1 .53.22l3.25 3.25c.141.14.22.331.22.53v9.25a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75zm1.5.75H5v3.75c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75V2.81l2.5 2.5v8.19h-11zm4 0h3v3h-3z", clipRule: "evenodd" })] }));
}
const SaveIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSaveIcon });
});
SaveIcon.displayName = 'SaveIcon';
export default SaveIcon;
//# sourceMappingURL=SaveIcon.js.map