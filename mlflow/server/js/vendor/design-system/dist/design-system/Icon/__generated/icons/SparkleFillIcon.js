import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSparkleFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M13.667 6.575c1.376.452 1.376 2.398 0 2.85l-2.472.813a1.5 1.5 0 0 0-.957.957l-.813 2.472c-.452 1.376-2.398 1.376-2.85 0l-.813-2.472a1.5 1.5 0 0 0-.956-.957l-2.473-.813c-1.376-.452-1.376-2.398 0-2.85l2.473-.813a1.5 1.5 0 0 0 .956-.956l.813-2.473c.452-1.376 2.398-1.376 2.85 0l.813 2.473a1.5 1.5 0 0 0 .957.956z", clipRule: "evenodd" }) }));
}
const SparkleFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSparkleFillIcon });
});
SparkleFillIcon.displayName = 'SparkleFillIcon';
export default SparkleFillIcon;
//# sourceMappingURL=SparkleFillIcon.js.map