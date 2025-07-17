import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgCloseSmallIcon(props) {
    return (_jsx("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M7.064 8 4 4.936 4.936 4 8 7.064 11.063 4l.937.936L8.937 8 12 11.063l-.937.937L8 8.937 4.936 12 4 11.063z", clipRule: "evenodd" }) }));
}
const CloseSmallIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgCloseSmallIcon });
});
CloseSmallIcon.displayName = 'CloseSmallIcon';
export default CloseSmallIcon;
//# sourceMappingURL=CloseSmallIcon.js.map