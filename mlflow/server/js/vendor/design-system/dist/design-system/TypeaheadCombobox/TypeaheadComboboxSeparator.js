import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useTypeaheadComboboxContext } from './hooks';
import { Separator } from '../_shared_/Combobox';
export const TypeaheadComboboxSeparator = (props) => {
    const { isInsideTypeaheadCombobox } = useTypeaheadComboboxContext();
    if (!isInsideTypeaheadCombobox) {
        throw new Error('`TypeaheadComboboxSeparator` must be used within `TypeaheadComboboxMenu`');
    }
    return _jsx(Separator, { ...props });
};
//# sourceMappingURL=TypeaheadComboboxSeparator.js.map