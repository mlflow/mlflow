import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMapIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m2.058 13.934 3.827-1.723 3.675 2.646.015.011a.75.75 0 0 0 .735.065l4.248-1.912a.75.75 0 0 0 .442-.684V2.75a.75.75 0 0 0-1.058-.684L10.115 3.79 6.44 1.143l-.015-.011a.75.75 0 0 0-.735-.065L1.442 2.979A.75.75 0 0 0 1 3.663v9.587a.75.75 0 0 0 1.058.684M2.5 4.148 5.25 2.91v7.942L2.5 12.09zm8.25 1v7.942l2.75-1.238V3.91zm-1.5-.134-2.5-1.8v7.772l2.5 1.8z", clipRule: "evenodd" }) }));
}
const MapIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMapIcon });
});
MapIcon.displayName = 'MapIcon';
export default MapIcon;
//# sourceMappingURL=MapIcon.js.map