import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgDagVerticalIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M4.5 2.75A.75.75 0 0 1 5.25 2h5.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-.564l2.571 2h2.493a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1-.75-.75v-3.5A.75.75 0 0 1 9.75 9h.564L8 7.2 5.686 9h.564a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75H.75a.75.75 0 0 1-.75-.75v-3.5A.75.75 0 0 1 .75 9h2.493l2.571-2H5.25a.75.75 0 0 1-.75-.75zM6 3.5v2h4v-2zm-4.5 7v2h4v-2zm9 0v2h4v-2z", clipRule: "evenodd" }) }));
}
const DagVerticalIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgDagVerticalIcon });
});
DagVerticalIcon.displayName = 'DagVerticalIcon';
export default DagVerticalIcon;
//# sourceMappingURL=DagVerticalIcon.js.map