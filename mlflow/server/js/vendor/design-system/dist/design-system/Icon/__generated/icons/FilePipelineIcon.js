import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFilePipelineIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 .75A.75.75 0 0 1 2.75 0h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53V8h-1.5V6H8.75A.75.75 0 0 1 8 5.25V1.5H3.5v12h6V15H2.75a.75.75 0 0 1-.75-.75zm7.5 1.81 1.94 1.94H9.5z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z", clipRule: "evenodd" })] }));
}
const FilePipelineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFilePipelineIcon });
});
FilePipelineIcon.displayName = 'FilePipelineIcon';
export default FilePipelineIcon;
//# sourceMappingURL=FilePipelineIcon.js.map