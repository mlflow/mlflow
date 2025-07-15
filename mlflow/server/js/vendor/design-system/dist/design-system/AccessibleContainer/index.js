import { Fragment as _Fragment, jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { visuallyHidden } from '../utils';
export function AccessibleContainer({ children, label }) {
    if (!label) {
        return _jsx(_Fragment, { children: children });
    }
    return (_jsxs("div", { css: { cursor: 'progress' }, children: [_jsx("span", { css: visuallyHidden, children: label }), _jsx("div", { "aria-hidden": true, children: children })] }));
}
//# sourceMappingURL=index.js.map