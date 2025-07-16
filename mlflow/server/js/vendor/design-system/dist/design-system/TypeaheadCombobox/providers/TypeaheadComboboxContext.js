import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext, useState } from 'react';
const typeaheadComboboxContextDefaults = {
    componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_providers_typeaheadcomboboxcontext.tsx_17',
    isInsideTypeaheadCombobox: false,
    multiSelect: false,
};
export const TypeaheadComboboxContext = createContext(typeaheadComboboxContextDefaults);
export const TypeaheadComboboxContextProvider = ({ children, value, }) => {
    const [inputWidth, setInputWidth] = useState();
    return (_jsx(TypeaheadComboboxContext.Provider, { value: { ...value, setInputWidth, inputWidth }, children: children }));
};
//# sourceMappingURL=TypeaheadComboboxContext.js.map