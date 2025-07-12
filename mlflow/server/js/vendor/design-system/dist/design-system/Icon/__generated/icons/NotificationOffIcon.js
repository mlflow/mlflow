import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgNotificationOffIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "m14.47 13.53-12-12-1 1L3.28 4.342A5 5 0 0 0 3 6v1.99c0 .674-.2 1.332-.573 1.892l-1.301 1.952A.75.75 0 0 0 1.75 13h3.5v.25a2.75 2.75 0 1 0 5.5 0V13h1.19l1.53 1.53zM13.038 8.5A3.4 3.4 0 0 1 13 7.99V6a5 5 0 0 0-7.965-4.026l1.078 1.078A3.5 3.5 0 0 1 11.5 6v1.99q0 .238.023.472l.038.038zM4.5 6q0-.21.024-.415L10.44 11.5H3.151l.524-.786A4.9 4.9 0 0 0 4.5 7.99zm2.25 7.25V13h2.5v.25a1.25 1.25 0 1 1-2.5 0", clipRule: "evenodd" }) }));
}
const NotificationOffIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgNotificationOffIcon });
});
NotificationOffIcon.displayName = 'NotificationOffIcon';
export default NotificationOffIcon;
//# sourceMappingURL=NotificationOffIcon.js.map