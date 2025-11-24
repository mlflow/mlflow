import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPipelineIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10.75 6.75A5.75 5.75 0 0 0 5 1H1.75a.75.75 0 0 0-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 0 1 .25.25v2.25A5.75 5.75 0 0 0 11 15h3.25a.75.75 0 0 0 .75-.75V10a.75.75 0 0 0-.75-.75H11a.25.25 0 0 1-.25-.25zM5.5 2.53a4.25 4.25 0 0 1 3.75 4.22V9a1.75 1.75 0 0 0 1.25 1.678v2.793A4.25 4.25 0 0 1 6.75 9.25V7A1.75 1.75 0 0 0 5.5 5.322zM4 2.5v2.75H2.5V2.5zm9.5 8.25H12v2.75h1.5z", clipRule: "evenodd" }) }));
}
const PipelineIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPipelineIcon });
});
PipelineIcon.displayName = 'PipelineIcon';
export default PipelineIcon;
//# sourceMappingURL=PipelineIcon.js.map