import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFileIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 1.75A.75.75 0 0 1 2.75 1h6a.75.75 0 0 1 .53.22l4.5 4.5c.141.14.22.331.22.53v9a.75.75 0 0 1-.75.75H2.75a.75.75 0 0 1-.75-.75zm1.5.75v12h9V7H8.75A.75.75 0 0 1 8 6.25V2.5zm6 1.06 1.94 1.94H9.5z", clipRule: "evenodd" }) }));
}
const FileIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFileIcon });
});
FileIcon.displayName = 'FileIcon';
export default FileIcon;
//# sourceMappingURL=FileIcon.js.map