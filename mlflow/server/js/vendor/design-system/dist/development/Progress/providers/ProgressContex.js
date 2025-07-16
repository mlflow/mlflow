import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext } from 'react';
const progressContextDefaults = {
    progress: 0,
};
export const ProgressContext = createContext(progressContextDefaults);
export const ProgressContextProvider = ({ children, value }) => {
    return _jsx(ProgressContext.Provider, { value: value, children: children });
};
//# sourceMappingURL=ProgressContex.js.map