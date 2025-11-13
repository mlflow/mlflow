import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgNotebookIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3 1.75A.75.75 0 0 1 3.75 1h10.5a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75H3.75a.75.75 0 0 1-.75-.75V12.5H1V11h2V8.75H1v-1.5h2V5H1V3.5h2zm1.5.75v11H6v-11zm3 0v11h6v-11z", clipRule: "evenodd" }) }));
}
const NotebookIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgNotebookIcon });
});
NotebookIcon.displayName = 'NotebookIcon';
export default NotebookIcon;
//# sourceMappingURL=NotebookIcon.js.map