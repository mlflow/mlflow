import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgSpeechBubbleQuestionMarkIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M6 1a6 6 0 0 0-6 6v.25a5.75 5.75 0 0 0 5 5.701v2.299a.75.75 0 0 0 1.28.53L9.06 13H10a6 6 0 0 0 0-12zM1.5 7A4.5 4.5 0 0 1 6 2.5h4a4.5 4.5 0 1 1 0 9H8.75a.75.75 0 0 0-.53.22L6.5 13.44v-1.19a.75.75 0 0 0-.75-.75A4.25 4.25 0 0 1 1.5 7.25zm8.707-1.689A2.25 2.25 0 0 1 8 8h-.75V6.5H8a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 9.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0", clipRule: "evenodd" }) }));
}
const SpeechBubbleQuestionMarkIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgSpeechBubbleQuestionMarkIcon });
});
SpeechBubbleQuestionMarkIcon.displayName = 'SpeechBubbleQuestionMarkIcon';
export default SpeechBubbleQuestionMarkIcon;
//# sourceMappingURL=SpeechBubbleQuestionMarkIcon.js.map