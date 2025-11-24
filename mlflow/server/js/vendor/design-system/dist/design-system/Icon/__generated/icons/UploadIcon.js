import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUploadIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M1 13.56h14v1.5H1zM12.53 5.53l-1.06 1.061-2.72-2.72v7.19h-1.5V3.87l-2.72 2.72-1.06-1.06L8 1z" }) }));
}
const UploadIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUploadIcon });
});
UploadIcon.displayName = 'UploadIcon';
export default UploadIcon;
//# sourceMappingURL=UploadIcon.js.map