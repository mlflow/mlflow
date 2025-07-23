import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFolderFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M.75 2a.75.75 0 0 0-.75.75v10.5c0 .414.336.75.75.75h14.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H7.81L6.617 2.805A2.75 2.75 0 0 0 4.672 2z" }) }));
}
const FolderFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFolderFillIcon });
});
FolderFillIcon.displayName = 'FolderFillIcon';
export default FolderFillIcon;
//# sourceMappingURL=FolderFillIcon.js.map