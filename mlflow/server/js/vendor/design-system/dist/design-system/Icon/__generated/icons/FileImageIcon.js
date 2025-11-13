import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFileImageIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V10h-1.5V7H8.75A.75.75 0 0 1 8 6.25V2.5H3.5V16H2zm7.5 1.81 1.94 1.94H9.5z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10.466 10a.75.75 0 0 0-.542.27l-3.75 4.5A.75.75 0 0 0 6.75 16h6.5a.75.75 0 0 0 .75-.75V13.5a.75.75 0 0 0-.22-.53l-2.75-2.75a.75.75 0 0 0-.564-.22m2.034 3.81v.69H8.351l2.2-2.639zM6.5 7.25a2.25 2.25 0 1 0 0 4.5 2.25 2.25 0 0 0 0-4.5M5.75 9.5a.75.75 0 1 1 1.5 0 .75.75 0 0 1-1.5 0", clipRule: "evenodd" })] }));
}
const FileImageIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFileImageIcon });
});
FileImageIcon.displayName = 'FileImageIcon';
export default FileImageIcon;
//# sourceMappingURL=FileImageIcon.js.map