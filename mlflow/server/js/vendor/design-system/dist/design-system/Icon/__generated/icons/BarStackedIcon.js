import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgBarStackedIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M6.25 1a.75.75 0 0 0-.75.75V7H2.75a.75.75 0 0 0-.75.75v6.5c0 .414.336.75.75.75h10.5a.75.75 0 0 0 .75-.75v-9.5a.75.75 0 0 0-.75-.75H10.5V1.75A.75.75 0 0 0 9.75 1zM9 8.5v5H7v-5zM9 7V2.5H7V7zm3.5 6.5h-2v-1.75h2zm-2-8v4.75h2V5.5zm-5 4.75V8.5h-2v1.75zm0 3.25v-1.75h-2v1.75z", clipRule: "evenodd" }) }));
}
const BarStackedIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgBarStackedIcon });
});
BarStackedIcon.displayName = 'BarStackedIcon';
export default BarStackedIcon;
//# sourceMappingURL=BarStackedIcon.js.map