import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgQuestionMarkFillIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16m2.207-10.189a2.25 2.25 0 0 1-1.457 2.56V9h-1.5V7.75A.75.75 0 0 1 8 7a.75.75 0 1 0-.75-.75h-1.5a2.25 2.25 0 0 1 4.457-.439M7.25 10.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0", clipRule: "evenodd" }) }));
}
const QuestionMarkFillIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgQuestionMarkFillIcon });
});
QuestionMarkFillIcon.displayName = 'QuestionMarkFillIcon';
export default QuestionMarkFillIcon;
//# sourceMappingURL=QuestionMarkFillIcon.js.map