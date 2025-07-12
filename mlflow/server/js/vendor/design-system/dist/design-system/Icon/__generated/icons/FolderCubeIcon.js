import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFolderCubeIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M12.314 8.069a.75.75 0 0 0-.628 0l-3.248 1.5-.017.007A.75.75 0 0 0 8 10.25v3.5c0 .293.17.558.436.681l3.246 1.498a.75.75 0 0 0 .636 0l3.246-1.498A.75.75 0 0 0 16 13.75v-3.5a.75.75 0 0 0-.436-.681m-2.104.681L12 9.576l-1.46.674 1.46.674zm1.04 1.172-1.75.808v1.848l1.75-.808zm-3.25.808-1.75-.808v1.848l1.75.808z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "m12.314 8.069 3.248 1.5Z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h5.268A3 3 0 0 1 6 13.68V12.5H1.5v-9h3.172c.331 0 .649.132.883.366L6.97 5.28c.14.141.331.22.53.22h7v1.362l1.5.692V4.75a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z", clipRule: "evenodd" })] }));
}
const FolderCubeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFolderCubeIcon });
});
FolderCubeIcon.displayName = 'FolderCubeIcon';
export default FolderCubeIcon;
//# sourceMappingURL=FolderCubeIcon.js.map