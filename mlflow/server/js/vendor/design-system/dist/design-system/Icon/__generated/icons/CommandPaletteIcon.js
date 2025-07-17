import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCommandPaletteIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", stroke: "currentColor", strokeWidth: 0.25, d: "M4.316 6.737H5.38v2.526H4.316c-1.346 0-2.441 1.057-2.441 2.415a2.441 2.441 0 1 0 4.882 0V10.62h2.486v1.057a2.441 2.441 0 1 0 4.882 0c0-1.358-1.094-2.415-2.44-2.415h-1.058V6.737h1.057c1.347 0 2.441-1.057 2.441-2.415a2.441 2.441 0 1 0-4.882 0V5.38H6.757V4.322a2.441 2.441 0 1 0-4.882 0c0 1.358 1.095 2.415 2.44 2.415ZM5.38 4.322v1.073H4.316A1.08 1.08 0 0 1 3.25 4.322c0-.585.485-1.064 1.065-1.064s1.065.479 1.065 1.064Zm7.368 0a1.08 1.08 0 0 1-1.065 1.073h-1.057V4.322c0-.587.479-1.064 1.057-1.064.58 0 1.066.479 1.066 1.064Zm-3.506 2.4v2.557H6.757V6.72zM3.251 11.67a1.08 1.08 0 0 1 1.065-1.073H5.38v1.073c0 .585-.486 1.064-1.065 1.064-.58 0-1.065-.479-1.065-1.064Zm7.376 0v-1.073h1.057a1.08 1.08 0 0 1 1.066 1.073c0 .585-.486 1.064-1.066 1.064a1.065 1.065 0 0 1-1.057-1.064Z" }) }));
}
const CommandPaletteIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCommandPaletteIcon });
});
CommandPaletteIcon.displayName = 'CommandPaletteIcon';
export default CommandPaletteIcon;
//# sourceMappingURL=CommandPaletteIcon.js.map