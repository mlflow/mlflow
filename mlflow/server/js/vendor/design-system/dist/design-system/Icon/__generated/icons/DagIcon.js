import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDagIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 1.75A.75.75 0 0 1 8.75 1h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5A.75.75 0 0 1 8 5.25v-1H5.5c-.69 0-1.25.56-1.25 1.25h2a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-2c0 .69.56 1.25 1.25 1.25H8v-1a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-1H5.5a2.75 2.75 0 0 1-2.75-2.75h-2A.75.75 0 0 1 0 9.75v-3.5a.75.75 0 0 1 .75-.75h2A2.75 2.75 0 0 1 5.5 2.75H8zm1.5.75v2h4v-2zM1.5 9V7h4v2zm8 4.5v-2h4v2z", clipRule: "evenodd" }) }));
}
const DagIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDagIcon });
});
DagIcon.displayName = 'DagIcon';
export default DagIcon;
//# sourceMappingURL=DagIcon.js.map