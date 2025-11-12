import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCloudUploadIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M8 2a4.75 4.75 0 0 0-4.606 3.586 4.251 4.251 0 0 0 .427 8.393A.8.8 0 0 0 4 14v-1.511a2.75 2.75 0 0 1 .077-5.484.75.75 0 0 0 .697-.657 3.25 3.25 0 0 1 6.476.402v.5c0 .414.336.75.75.75h.25a2.25 2.25 0 1 1-.188 4.492L12 12.49V14l.077-.004q.086.004.173.004a3.75 3.75 0 0 0 .495-7.468A4.75 4.75 0 0 0 8 2" }), _jsx("path", { fill: "currentColor", d: "m8.75 8.81 2.22 2.22 1.06-1.06L8 5.94 3.97 9.97l1.06 1.06 2.22-2.22V14h1.5z" })] }));
}
const CloudUploadIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCloudUploadIcon });
});
CloudUploadIcon.displayName = 'CloudUploadIcon';
export default CloudUploadIcon;
//# sourceMappingURL=CloudUploadIcon.js.map