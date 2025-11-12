import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCreditCardIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M13 9H9v1.5h4z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.75 2A1.75 1.75 0 0 0 0 3.75v8.5C0 13.216.784 14 1.75 14h12.5A1.75 1.75 0 0 0 16 12.25v-8.5A1.75 1.75 0 0 0 14.25 2zM1.5 3.75a.25.25 0 0 1 .25-.25h12.5a.25.25 0 0 1 .25.25V5.5h-13zM1.5 7h13v5.25a.25.25 0 0 1-.25.25H1.75a.25.25 0 0 1-.25-.25z", clipRule: "evenodd" })] }));
}
const CreditCardIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCreditCardIcon });
});
CreditCardIcon.displayName = 'CreditCardIcon';
export default CreditCardIcon;
//# sourceMappingURL=CreditCardIcon.js.map