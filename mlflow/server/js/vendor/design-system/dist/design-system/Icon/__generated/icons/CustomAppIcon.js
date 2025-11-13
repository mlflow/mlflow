import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCustomAppIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M14.648 10.625a.75.75 0 0 1-.273 1.023l-6 3.5a.75.75 0 0 1-.756 0l-6-3.5a.75.75 0 0 1 .755-1.296L8 13.632l5.626-3.28a.75.75 0 0 1 1.023.273m-1.023-3.273L8 10.632l-5.622-3.28a.75.75 0 0 0-.753 1.296l6 3.5a.75.75 0 0 0 .756 0l6-3.5a.75.75 0 0 0-.756-1.296M1.25 5a.75.75 0 0 1 .375-.648l6-3.5a.75.75 0 0 1 .756 0l6 3.5a.75.75 0 0 1 0 1.296l-6 3.5a.75.75 0 0 1-.756 0l-6-3.5A.75.75 0 0 1 1.25 5m2.239 0L8 7.632 12.511 5 8 2.368z" }) }));
}
const CustomAppIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCustomAppIcon });
});
CustomAppIcon.displayName = 'CustomAppIcon';
export default CustomAppIcon;
//# sourceMappingURL=CustomAppIcon.js.map