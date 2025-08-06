import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgMinusCircleFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m3.5-7.25h-7v-1.5h7z", clipRule: "evenodd" }) }));
}
const MinusCircleFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgMinusCircleFillIcon });
});
MinusCircleFillIcon.displayName = 'MinusCircleFillIcon';
export default MinusCircleFillIcon;
//# sourceMappingURL=MinusCircleFillIcon.js.map