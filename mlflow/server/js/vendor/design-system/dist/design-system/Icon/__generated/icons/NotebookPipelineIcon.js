import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgNotebookPipelineIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75V8h-1.5V2.5h-6v11h2V15H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2zm1.5.75v11H6v-11z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M9.25 8.5a.75.75 0 0 0-.75.75v2.5c0 .414.336.75.75.75h.785a3.5 3.5 0 0 0 3.465 3h1.25a.75.75 0 0 0 .75-.75v-2.5a.75.75 0 0 0-.75-.75h-.785a3.5 3.5 0 0 0-3.465-3zM10 11v-1h.5a2 2 0 0 1 2 2 1 1 0 0 0 1 1h.5v1h-.5a2 2 0 0 1-2-2 1 1 0 0 0-1-1z", clipRule: "evenodd" })] }));
}
const NotebookPipelineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgNotebookPipelineIcon });
});
NotebookPipelineIcon.displayName = 'NotebookPipelineIcon';
export default NotebookPipelineIcon;
//# sourceMappingURL=NotebookPipelineIcon.js.map