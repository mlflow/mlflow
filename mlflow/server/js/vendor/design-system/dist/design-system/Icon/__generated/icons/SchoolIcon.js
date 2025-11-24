import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSchoolIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M16 7a.75.75 0 0 0-.37-.647l-7.25-4.25a.75.75 0 0 0-.76 0L.37 6.353a.75.75 0 0 0 0 1.294L3 9.188V12a.75.75 0 0 0 .4.663l4.25 2.25a.75.75 0 0 0 .7 0l4.25-2.25A.75.75 0 0 0 13 12V9.188l1.5-.879V12H16zm-7.62 4.897 3.12-1.83v1.481L8 13.401l-3.5-1.853v-1.48l3.12 1.829a.75.75 0 0 0 .76 0M8 3.619 2.233 7 8 10.38 13.767 7z", clipRule: "evenodd" }) }));
}
const SchoolIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSchoolIcon });
});
SchoolIcon.displayName = 'SchoolIcon';
export default SchoolIcon;
//# sourceMappingURL=SchoolIcon.js.map