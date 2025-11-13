import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgColorFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.5 1v1.59l4.88 4.88a.75.75 0 0 1 0 1.06l-4.242 4.243a2.75 2.75 0 0 1-3.89 0l-2.421-2.422a2.75 2.75 0 0 1 0-3.889L6 2.29V1zM6 8V4.41L2.888 7.524a1.25 1.25 0 0 0 0 1.768l2.421 2.421a1.25 1.25 0 0 0 1.768 0L10.789 8 7.5 4.71V8zm7.27 1.51a.76.76 0 0 0-1.092.001 8.5 8.5 0 0 0-1.216 1.636c-.236.428-.46.953-.51 1.501-.054.576.083 1.197.587 1.701a2.385 2.385 0 0 0 3.372 0c.505-.504.644-1.126.59-1.703-.05-.55-.274-1.075-.511-1.503a8.5 8.5 0 0 0-1.22-1.633m-.995 2.363c.138-.25.3-.487.451-.689.152.201.313.437.452.687.19.342.306.657.33.913.02.228-.03.377-.158.505a.885.885 0 0 1-1.25 0c-.125-.125-.176-.272-.155-.501.024-.256.14-.572.33-.915", clipRule: "evenodd" }) }));
}
const ColorFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgColorFillIcon });
});
ColorFillIcon.displayName = 'ColorFillIcon';
export default ColorFillIcon;
//# sourceMappingURL=ColorFillIcon.js.map