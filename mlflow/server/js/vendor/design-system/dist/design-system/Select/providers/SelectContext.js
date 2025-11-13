import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext } from 'react';
const selectContextDefaults = {
    isSelect: false,
};
export const SelectContext = createContext(selectContextDefaults);
export const SelectContextProvider = ({ children, value }) => {
    return _jsx(SelectContext.Provider, { value: value, children: children });
};
//# sourceMappingURL=SelectContext.js.map