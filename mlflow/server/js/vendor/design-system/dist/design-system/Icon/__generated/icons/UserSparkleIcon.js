import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { forwardRef } from 'react';
import { Icon } from '../../Icon';
function SvgUserSparkleIcon(props) {
    return (_jsxs("svg", { xmlns: "http://www.w3.org/2000/svg", width: "1em", height: "1em", fill: "none", viewBox: "0 0 16 16", ...props, children: [_jsx("path", { fill: "currentColor", d: "M8 1c.664 0 1.282.2 1.797.542l-.014.072-.062.357-.357.062c-.402.07-.765.245-1.06.493a1.75 1.75 0 1 0 0 3.447c.295.25.658.424 1.06.494l.357.062.062.357.014.072A3.25 3.25 0 1 1 8 1" }), _jsx("path", { fill: "currentColor", d: "M9.59 4.983A.75.75 0 0 1 9.62 3.51l.877-.152a.75.75 0 0 0 .61-.61l.153-.878a.75.75 0 0 1 1.478 0l.152.877a.75.75 0 0 0 .61.61l.878.153a.75.75 0 0 1 0 1.478l-.877.152a.75.75 0 0 0-.61.61l-.153.878a.75.75 0 0 1-1.478 0l-.152-.877a.75.75 0 0 0-.61-.61l-.878-.153z" }), _jsx("path", { fill: "currentColor", fillRule: "evenodd", d: "M1.164 12.287A8.74 8.74 0 0 1 8 9a8.74 8.74 0 0 1 6.836 3.287.75.75 0 0 1 .164.469v1.494a.75.75 0 0 1-.75.75H1.75a.75.75 0 0 1-.75-.75v-1.494a.75.75 0 0 1 .164-.469m1.336.74v.473h11v-.474A7.23 7.23 0 0 0 8 10.5c-2.2 0-4.17.978-5.5 2.526", clipRule: "evenodd" })] }));
}
const UserSparkleIcon = forwardRef((props, forwardedRef) => {
    return _jsx(Icon, { ref: forwardedRef, ...props, component: SvgUserSparkleIcon });
});
UserSparkleIcon.displayName = 'UserSparkleIcon';
export default UserSparkleIcon;
//# sourceMappingURL=UserSparkleIcon.js.map