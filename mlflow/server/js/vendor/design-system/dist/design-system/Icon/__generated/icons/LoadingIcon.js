import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgLoadingIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 24 24", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M23.212 12a.79.79 0 0 1-.789-.788 9.6 9.6 0 0 0-.757-3.751 9.66 9.66 0 0 0-5.129-5.129 9.6 9.6 0 0 0-3.749-.755.788.788 0 0 1 0-1.577c1.513 0 2.983.296 4.365.882a11.1 11.1 0 0 1 3.562 2.403 11.157 11.157 0 0 1 3.283 7.927.785.785 0 0 1-.786.788", clipRule: "evenodd" }) }));
}
const LoadingIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgLoadingIcon });
});
LoadingIcon.displayName = 'LoadingIcon';
export default LoadingIcon;
//# sourceMappingURL=LoadingIcon.js.map