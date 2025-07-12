import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext } from 'react';
export const DialogComboboxOptionListContext = createContext({
    isInsideDialogComboboxOptionList: false,
    lookAhead: '',
    setLookAhead: () => { },
});
export const DialogComboboxOptionListContextProvider = ({ children, value, }) => {
    return _jsx(DialogComboboxOptionListContext.Provider, { value: value, children: children });
};
//# sourceMappingURL=DialogComboboxOptionListContext.js.map