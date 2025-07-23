import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgQuestionMarkIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0M10.079 7.111A2.25 2.25 0 1 0 5.75 6.25h1.5A.75.75 0 1 1 8 7a.75.75 0 0 0-.75.75V9h1.5v-.629a2.25 2.25 0 0 0 1.329-1.26" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13", clipRule: "evenodd" })] }));
}
const QuestionMarkIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgQuestionMarkIcon });
});
QuestionMarkIcon.displayName = 'QuestionMarkIcon';
export default QuestionMarkIcon;
//# sourceMappingURL=QuestionMarkIcon.js.map