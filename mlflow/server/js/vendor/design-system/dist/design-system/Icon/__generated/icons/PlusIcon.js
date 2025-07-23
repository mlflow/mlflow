import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgPlusIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.25 7.25V1h1.5v6.25H15v1.5H8.75V15h-1.5V8.75H1v-1.5z", clipRule: "evenodd" }) }));
}
const PlusIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgPlusIcon });
});
PlusIcon.displayName = 'PlusIcon';
export default PlusIcon;
//# sourceMappingURL=PlusIcon.js.map