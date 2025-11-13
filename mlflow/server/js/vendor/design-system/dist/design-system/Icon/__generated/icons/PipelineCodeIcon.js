import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPipelineCodeIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M10.75 6.75A5.75 5.75 0 0 0 5 1H1.75a.75.75 0 0 0-.75.75V6c0 .414.336.75.75.75H5a.25.25 0 0 1 .25.25v2.25c0 1.338.457 2.57 1.223 3.546l1.072-1.071A4.23 4.23 0 0 1 6.75 9.25V7A1.75 1.75 0 0 0 5.5 5.322V2.53A4.25 4.25 0 0 1 9.25 6.75V9c0 .295.073.573.202.817L10.75 8.52zM4 2.5v2.75H2.5V2.5z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "m13.319 10.66 2.608 2.61-2.608 2.609-1.061-1.061 1.548-1.548-1.548-1.549zM10.68 10.66l-2.61 2.61 2.61 2.609 1.06-1.06-1.549-1.55 1.549-1.548z" })] }));
}
const PipelineCodeIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPipelineCodeIcon });
});
PipelineCodeIcon.displayName = 'PipelineCodeIcon';
export default PipelineCodeIcon;
//# sourceMappingURL=PipelineCodeIcon.js.map