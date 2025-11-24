import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgFilterIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1 1.75A.75.75 0 0 1 1.75 1h12.5a.75.75 0 0 1 .75.75V4a.75.75 0 0 1-.22.53L10 9.31v4.94a.75.75 0 0 1-.75.75h-2.5a.75.75 0 0 1-.75-.75V9.31L1.22 4.53A.75.75 0 0 1 1 4zm1.5.75v1.19l4.78 4.78c.141.14.22.331.22.53v4.5h1V9a.75.75 0 0 1 .22-.53l4.78-4.78V2.5z", clipRule: "evenodd" }) }));
}
const FilterIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgFilterIcon });
});
FilterIcon.displayName = 'FilterIcon';
export default FilterIcon;
//# sourceMappingURL=FilterIcon.js.map