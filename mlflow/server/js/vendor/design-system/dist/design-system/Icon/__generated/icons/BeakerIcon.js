import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBeakerIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.75 1a.75.75 0 0 0-.75.75v6.089c0 .38-.173.739-.47.976l-2.678 2.143A2.27 2.27 0 0 0 3.27 15h9.46a2.27 2.27 0 0 0 1.418-4.042L11.47 8.815A1.25 1.25 0 0 1 11 7.839V1.75a.75.75 0 0 0-.75-.75zm.75 6.839V2.5h3v5.339c0 .606.2 1.188.559 1.661H5.942A2.75 2.75 0 0 0 6.5 7.839M4.2 11 2.79 12.13a.77.77 0 0 0 .48 1.37h9.461a.77.77 0 0 0 .481-1.37L11.8 11z", clipRule: "evenodd" }) }));
}
const BeakerIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBeakerIcon });
});
BeakerIcon.displayName = 'BeakerIcon';
export default BeakerIcon;
//# sourceMappingURL=BeakerIcon.js.map