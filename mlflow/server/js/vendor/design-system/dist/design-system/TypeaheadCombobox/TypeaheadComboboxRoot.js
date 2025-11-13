import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { useFloating, flip, shift, autoUpdate, offset, useMergeRefs } from '@floating-ui/react';
import { forwardRef } from 'react';
import { TypeaheadComboboxContextProvider } from './providers/TypeaheadComboboxContext';
import { DesignSystemEventProviderComponentTypes } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';
export const TypeaheadComboboxRoot = forwardRef(({ comboboxState, multiSelect = false, children, ...props }, ref) => {
    const { classNamePrefix } = useDesignSystemTheme();
    const { refs, floatingStyles } = useFloating({
        whileElementsMounted: autoUpdate,
        middleware: [offset(4), flip(), shift()],
        placement: 'bottom-start',
    });
    const { elementRef: typeaheadComboboxRootRef } = useNotifyOnFirstView({
        onView: comboboxState.onView,
        value: comboboxState.firstOnViewValue,
    });
    const mergedRef = useMergeRefs([ref, typeaheadComboboxRootRef]);
    return (_jsx(TypeaheadComboboxContextProvider, { value: {
            componentId: comboboxState.componentId,
            multiSelect,
            isInsideTypeaheadCombobox: true,
            floatingUiRefs: refs,
            floatingStyles: floatingStyles,
        }, children: _jsx("div", { ...comboboxState.getComboboxProps({}, { suppressRefError: true }), className: `${classNamePrefix}-typeahead-combobox`, css: { display: 'inline-block', width: '100%' }, ...props, ref: mergedRef, "data-component-type": DesignSystemEventProviderComponentTypes.TypeaheadCombobox, "data-component-id": comboboxState.componentId, children: children }) }));
});
export default TypeaheadComboboxRoot;
//# sourceMappingURL=TypeaheadComboboxRoot.js.map