import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDownloadIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 13.5h14V15H1zM12.53 6.53l-1.06-1.06-2.72 2.72V1h-1.5v7.19L4.53 5.47 3.47 6.53 8 11.06z" }) }));
}
const DownloadIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDownloadIcon });
});
DownloadIcon.displayName = 'DownloadIcon';
export default DownloadIcon;
//# sourceMappingURL=DownloadIcon.js.map