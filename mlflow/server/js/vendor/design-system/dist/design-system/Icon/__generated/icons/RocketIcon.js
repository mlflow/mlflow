import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgRocketIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M13.25 2a.75.75 0 0 1 .75.75v.892a8.75 8.75 0 0 1-3.07 6.656h.015v.626a4.75 4.75 0 0 1-2.017 3.884l-1.496 1.053a.75.75 0 0 1-1.163-.446l-.72-3.148-1.814-1.815L.589 9.75a.75.75 0 0 1-.451-1.162L1.193 7.08a4.75 4.75 0 0 1 3.891-2.025h.618v.015A8.75 8.75 0 0 1 12.358 2zM7.105 12.341l.377 1.65.583-.41a3.25 3.25 0 0 0 1.353-2.245q-.405.22-.837.397zM4.267 7.419l-.61 1.48L2.01 8.53l.413-.589a3.25 3.25 0 0 1 2.242-1.358q-.22.404-.397.836M12.5 3.5h-.142a7.2 7.2 0 0 0-2.754.543l2.353 2.353a7.2 7.2 0 0 0 .543-2.754zM5.654 7.99a7.24 7.24 0 0 1 2.576-3.2l2.98 2.98a7.24 7.24 0 0 1-3.2 2.576l-1.601.66L4.995 9.59z", clipRule: "evenodd" }), _jsx("path", { fill: "currentColor", d: "m2.22 10.72-.122.121A3.75 3.75 0 0 0 1 13.493v.757c0 .414.336.75.75.75h.757a3.75 3.75 0 0 0 2.652-1.098l.121-.122-1.06-1.06-.122.121a2.25 2.25 0 0 1-1.59.659H2.5v-.007c0-.597.237-1.17.659-1.591l.121-.122z" })] }));
}
const RocketIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgRocketIcon });
});
RocketIcon.displayName = 'RocketIcon';
export default RocketIcon;
//# sourceMappingURL=RocketIcon.js.map