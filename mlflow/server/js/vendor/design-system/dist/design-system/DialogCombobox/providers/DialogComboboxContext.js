import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext } from 'react';
const dialogComboboxContextDefaults = {
    componentId: 'codegen_design-system_src_design-system_dialogcombobox_providers_dialogcomboboxcontext.tsx_27',
    id: '',
    label: '',
    value: [],
    isInsideDialogCombobox: false,
    multiSelect: false,
    setValue: () => { },
    setIsControlled: () => { },
    stayOpenOnSelection: false,
    isOpen: false,
    setIsOpen: () => { },
    contentWidth: undefined,
    setContentWidth: () => { },
    textOverflowMode: 'multiline',
    setTextOverflowMode: () => { },
    scrollToSelectedElement: true,
    rememberLastScrollPosition: false,
    disableMouseOver: false,
    setDisableMouseOver: () => { },
    onView: () => { },
};
export const DialogComboboxContext = createContext(dialogComboboxContextDefaults);
export const DialogComboboxContextProvider = ({ children, value, }) => {
    return _jsx(DialogComboboxContext.Provider, { value: value, children: children });
};
//# sourceMappingURL=DialogComboboxContext.js.map