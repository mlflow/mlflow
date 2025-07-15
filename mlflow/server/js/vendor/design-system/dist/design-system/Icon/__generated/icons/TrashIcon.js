import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgTrashIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M6 0a.75.75 0 0 0-.712.513L4.46 3H1v1.5h1.077l1.177 10.831A.75.75 0 0 0 4 16h8a.75.75 0 0 0 .746-.669L13.923 4.5H15V3h-3.46L10.713.513A.75.75 0 0 0 10 0zm3.96 3-.5-1.5H6.54L6.04 3zM3.585 4.5l1.087 10h6.654l1.087-10z", clipRule: "evenodd" }) }));
}
const TrashIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgTrashIcon });
});
TrashIcon.displayName = 'TrashIcon';
export default TrashIcon;
//# sourceMappingURL=TrashIcon.js.map