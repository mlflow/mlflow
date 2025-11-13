import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlugIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m14.168 2.953.893-.892L14 1l-.893.893a4 4 0 0 0-5.077.48l-.884.884a.75.75 0 0 0 0 1.061l4.597 4.596a.75.75 0 0 0 1.06 0l.884-.884a4 4 0 0 0 .48-5.077M12.627 6.97l-.354.353-3.536-3.535.354-.354a2.5 2.5 0 1 1 3.536 3.536M7.323 10.152 5.91 8.737l1.414-1.414-1.06-1.06-1.415 1.414-.53-.53a.75.75 0 0 0-1.06 0l-.885.883a4 4 0 0 0-.48 5.077L1 14l1.06 1.06.893-.892a4 4 0 0 0 5.077-.48l.884-.885a.75.75 0 0 0 0-1.06l-.53-.53 1.414-1.415-1.06-1.06zm-3.889 2.475a2.5 2.5 0 0 0 3.536 0l.353-.354-3.535-3.536-.354.354a2.5 2.5 0 0 0 0 3.536", clipRule: "evenodd" }) }));
}
const PlugIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlugIcon });
});
PlugIcon.displayName = 'PlugIcon';
export default PlugIcon;
//# sourceMappingURL=PlugIcon.js.map