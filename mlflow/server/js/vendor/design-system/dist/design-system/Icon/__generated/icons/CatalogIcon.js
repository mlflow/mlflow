import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCatalogIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M14 .75a.75.75 0 0 0-.75-.75H4.5A2.5 2.5 0 0 0 2 2.5v10.75A2.75 2.75 0 0 0 4.75 16h8.5a.75.75 0 0 0 .75-.75zM3.5 4.792v8.458c0 .69.56 1.25 1.25 1.25h7.75V5h-8c-.356 0-.694-.074-1-.208m9-1.292v-2h-8a1 1 0 0 0 0 2z", clipRule: "evenodd" }) }));
}
const CatalogIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCatalogIcon });
});
CatalogIcon.displayName = 'CatalogIcon';
export default CatalogIcon;
//# sourceMappingURL=CatalogIcon.js.map