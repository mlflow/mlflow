import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgAssistantIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", d: "M11.28 11.03H4.73v-1.7h6.55zm-2.8-4.7H4.73v1.7h3.75zM15.79 8h-1.7a6.09 6.09 0 0 1-6.08 6.08H3.12l.58-.58c.33-.33.33-.87 0-1.2A6.04 6.04 0 0 1 1.92 8 6.09 6.09 0 0 1 8 1.92V.22C3.71.22.22 3.71.22 8c0 1.79.6 3.49 1.71 4.87L.47 14.33c-.24.24-.32.61-.18.93.13.32.44.52.79.52h6.93c4.29 0 7.78-3.49 7.78-7.78m-.62-3.47c.4-.15.4-.72 0-.88l-1.02-.38c-.73-.28-1.31-.85-1.58-1.58L12.19.67c-.08-.2-.26-.3-.44-.3s-.36.1-.44.3l-.38 1.02c-.28.73-.85 1.31-1.58 1.58l-1.02.38c-.4.15-.4.72 0 .88l1.02.38c.73.28 1.31.85 1.58 1.58l.38 1.02c.08.2.26.3.44.3s.36-.1.44-.3l.38-1.02c.28-.73.85-1.31 1.58-1.58z" }) }));
}
const AssistantIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgAssistantIcon });
});
AssistantIcon.displayName = 'AssistantIcon';
export default AssistantIcon;
//# sourceMappingURL=AssistantIcon.js.map