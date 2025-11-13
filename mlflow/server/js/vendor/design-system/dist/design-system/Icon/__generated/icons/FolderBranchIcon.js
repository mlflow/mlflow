import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFolderBranchIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 2.75A.75.75 0 0 1 .75 2h3.922c.729 0 1.428.29 1.944.805L7.811 4h7.439a.75.75 0 0 1 .75.75V8h-1.5V5.5h-7a.75.75 0 0 1-.53-.22L5.555 3.866a1.25 1.25 0 0 0-.883-.366H1.5v9H5V14H.75a.75.75 0 0 1-.75-.75zM9 8.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1M7 9a2 2 0 1 1 3.778.917c.376.58.888 1.031 1.414 1.227a2 2 0 1 1-.072 1.54c-.977-.207-1.795-.872-2.37-1.626v1.087a2 2 0 1 1-1.5 0v-1.29A2 2 0 0 1 7 9m7 2.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1m-5 2a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1", clipRule: "evenodd" }) }));
}
const FolderBranchIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFolderBranchIcon });
});
FolderBranchIcon.displayName = 'FolderBranchIcon';
export default FolderBranchIcon;
//# sourceMappingURL=FolderBranchIcon.js.map