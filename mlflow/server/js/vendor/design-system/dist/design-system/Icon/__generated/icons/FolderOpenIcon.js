import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFolderOpenIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h5.439a.75.75 0 0 1 .75.75V7h1.25a.75.75 0 0 1 .658 1.11l-3 5.5a.75.75 0 0 1-.658.39H.75a.747.747 0 0 1-.75-.75zm1.5 7.559L3.092 7.39A.75.75 0 0 1 3.75 7h8.75V5.5h-5a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5z", clipRule: "evenodd" }) }));
}
const FolderOpenIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFolderOpenIcon });
});
FolderOpenIcon.displayName = 'FolderOpenIcon';
export default FolderOpenIcon;
//# sourceMappingURL=FolderOpenIcon.js.map