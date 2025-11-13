import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCloudIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M3.394 5.586a4.752 4.752 0 0 1 9.351.946 3.75 3.75 0 0 1-.668 7.464L12 14H4a.8.8 0 0 1-.179-.021 4.25 4.25 0 0 1-.427-8.393m.72 6.914h7.762a.8.8 0 0 1 .186-.008q.092.008.188.008a2.25 2.25 0 0 0 0-4.5H12a.75.75 0 0 1-.75-.75v-.5a3.25 3.25 0 0 0-6.475-.402.75.75 0 0 1-.698.657 2.75 2.75 0 0 0-.024 5.488z", clipRule: "evenodd" }) }));
}
const CloudIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCloudIcon });
});
CloudIcon.displayName = 'CloudIcon';
export default CloudIcon;
//# sourceMappingURL=CloudIcon.js.map