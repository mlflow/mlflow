import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgZoomToFitIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "m2.5 3.56 2.97 2.97 1.06-1.06L3.56 2.5H6V1H1v5h1.5zM10.53 6.53l2.97-2.97V6H15V1h-5v1.5h2.44L9.47 5.47zM9.47 10.53l2.97 2.97H10V15h5v-5h-1.5v2.44l-2.97-2.97zM5.47 9.47 2.5 12.44V10H1v5h5v-1.5H3.56l2.97-2.97z" }) }));
}
const ZoomToFitIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgZoomToFitIcon });
});
ZoomToFitIcon.displayName = 'ZoomToFitIcon';
export default ZoomToFitIcon;
//# sourceMappingURL=ZoomToFitIcon.js.map