import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { TableRowAction } from './TableRowAction';
import { visuallyHidden } from '../utils';
export const TableRowActionHeader = ({ children }) => {
    return (_jsx(TableRowAction, { children: _jsx("span", { css: visuallyHidden, children: children }) }));
};
//# sourceMappingURL=TableRowActionHeader.js.map