import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBugIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M5.5 5a2.5 2.5 0 1 1 4.792 1H5.708A2.5 2.5 0 0 1 5.5 5M4.13 6.017a4 4 0 1 1 7.74 0l.047.065L14 4l1.06 1.06-2.41 2.412q.178.493.268 1.028H16V10h-3.02a6 6 0 0 1-.33 1.528l2.41 2.412L14 15l-2.082-2.082C11.002 14.187 9.588 15 8 15c-1.587 0-3.002-.813-3.918-2.082L2 15 .94 13.94l2.41-2.412A6 6 0 0 1 3.02 10H0V8.5h3.082q.09-.535.269-1.028L.939 5.061 2 4l2.082 2.081zm.812 1.538A4.4 4.4 0 0 0 4.5 9.5c0 2.347 1.698 4 3.5 4s3.5-1.653 3.5-4c0-.713-.163-1.375-.442-1.945z", clipRule: "evenodd" }) }));
}
const BugIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBugIcon });
});
BugIcon.displayName = 'BugIcon';
export default BugIcon;
//# sourceMappingURL=BugIcon.js.map