import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgWrenchIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M14.367 3.29a.75.75 0 0 1 .547.443 5.001 5.001 0 0 1-6.072 6.736l-3.187 3.186a2.341 2.341 0 0 1-3.31-3.31L5.53 7.158a5.001 5.001 0 0 1 6.736-6.072.75.75 0 0 1 .237 1.22L10.5 4.312V5.5h1.19l2.003-2.004a.75.75 0 0 1 .674-.206m-.56 2.214L12.53 6.78A.75.75 0 0 1 12 7H9.75A.75.75 0 0 1 9 6.25V4a.75.75 0 0 1 .22-.53l1.275-1.276a3.501 3.501 0 0 0-3.407 4.865.75.75 0 0 1-.16.823l-3.523 3.523a.84.84 0 1 0 1.19 1.19L8.118 9.07a.75.75 0 0 1 .823-.16 3.5 3.5 0 0 0 4.865-3.407", clipRule: "evenodd" }) }));
}
const WrenchIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgWrenchIcon });
});
WrenchIcon.displayName = 'WrenchIcon';
export default WrenchIcon;
//# sourceMappingURL=WrenchIcon.js.map