import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgGearFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.966 0q-.51 0-1.005.063a.75.75 0 0 0-.62.51l-.639 1.946q-.315.13-.61.294L3.172 2.1a.75.75 0 0 0-.784.165c-.481.468-.903.996-1.255 1.572a.75.75 0 0 0 .013.802l1.123 1.713a6 6 0 0 0-.15.66L.363 8.07a.75.75 0 0 0-.36.716c.067.682.22 1.34.447 1.962a.75.75 0 0 0 .635.489l2.042.19q.195.276.422.529l-.27 2.032a.75.75 0 0 0 .336.728 8 8 0 0 0 1.812.874.75.75 0 0 0 .778-.192l1.422-1.478a6 6 0 0 0 .677 0l1.422 1.478a.75.75 0 0 0 .778.192 8 8 0 0 0 1.812-.874.75.75 0 0 0 .335-.728l-.269-2.032a6 6 0 0 0 .422-.529l2.043-.19a.75.75 0 0 0 .634-.49c.228-.621.38-1.279.447-1.961a.75.75 0 0 0-.36-.716l-1.756-1.056a6 6 0 0 0-.15-.661l1.123-1.713a.75.75 0 0 0 .013-.802 8 8 0 0 0-1.255-1.572.75.75 0 0 0-.784-.165l-1.92.713q-.295-.163-.61-.294L9.589.573a.75.75 0 0 0-.619-.51A8 8 0 0 0 7.965 0m.018 10.25a2.25 2.25 0 1 0 0-4.5 2.25 2.25 0 0 0 0 4.5", clipRule: "evenodd" }) }));
}
const GearFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgGearFillIcon });
});
GearFillIcon.displayName = 'GearFillIcon';
export default GearFillIcon;
//# sourceMappingURL=GearFillIcon.js.map