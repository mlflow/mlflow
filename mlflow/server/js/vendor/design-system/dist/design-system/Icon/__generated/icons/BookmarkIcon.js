import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBookmarkIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M2 .75A.75.75 0 0 1 2.75 0h10.5a.75.75 0 0 1 .75.75v14.5a.75.75 0 0 1-1.28.53L8 11.06l-4.72 4.72A.75.75 0 0 1 2 15.25zm1.5.75v11.94l3.97-3.97a.75.75 0 0 1 1.06 0l3.97 3.97V1.5z", clipRule: "evenodd" }) }));
}
const BookmarkIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBookmarkIcon });
});
BookmarkIcon.displayName = 'BookmarkIcon';
export default BookmarkIcon;
//# sourceMappingURL=BookmarkIcon.js.map