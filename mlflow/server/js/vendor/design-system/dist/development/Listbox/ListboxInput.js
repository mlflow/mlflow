import { jsx as _jsx } from "@emotion/react/jsx-runtime";
/** @jsxImportSource @emotion/react */
import { useCallback } from 'react';
import { useListboxContext } from './ListboxRoot';
import { SearchIcon, useDesignSystemTheme } from '../../design-system';
import { Input } from '../../design-system/Input';
export const ListboxInput = ({ value, onChange, placeholder, 'aria-controls': ariaControls, 'aria-activedescendant': ariaActiveDescendant, className, options, }) => {
    const { handleKeyNavigation } = useListboxContext();
    const designSystemTheme = useDesignSystemTheme();
    const handleChange = useCallback((event) => {
        onChange(event.target.value);
    }, [onChange]);
    const handleKeyDown = useCallback((event) => {
        // Only handle navigation keys if there are options
        if (options.length > 0 && ['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter'].includes(event.key)) {
            handleKeyNavigation(event, options);
        }
    }, [handleKeyNavigation, options]);
    return (_jsx("div", { css: {
            position: 'sticky',
            top: 0,
            background: designSystemTheme.theme.colors.backgroundPrimary,
            zIndex: designSystemTheme.theme.options.zIndexBase + 1,
        }, children: _jsx(Input, { componentId: "listbox-filter-input", role: "combobox", "aria-controls": ariaControls, "aria-activedescendant": ariaActiveDescendant, "aria-expanded": "true", "aria-autocomplete": "list", value: value, onChange: handleChange, onKeyDown: handleKeyDown, placeholder: placeholder, prefix: _jsx(SearchIcon, {}), className: className, allowClear: true }) }));
};
//# sourceMappingURL=ListboxInput.js.map