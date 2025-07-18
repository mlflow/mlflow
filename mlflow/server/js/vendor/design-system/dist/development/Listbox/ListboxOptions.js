import { jsx as _jsx } from "@emotion/react/jsx-runtime";
/** @jsxImportSource @emotion/react */
import { css, useTheme } from '@emotion/react';
import { useCallback, useEffect, useRef } from 'react';
import { useListboxContext } from './ListboxRoot';
import { getComboboxOptionItemWrapperStyles } from '../../design-system/_shared_/Combobox/styles';
export const ListboxOptions = ({ options, onSelect, onHighlight, className }) => {
    const theme = useTheme();
    const { listboxId, selectedValue, setSelectedValue, highlightedValue, handleKeyNavigation } = useListboxContext();
    const listboxRef = useRef(null);
    const handleKeyDown = useCallback((event) => {
        handleKeyNavigation(event, options);
    }, [handleKeyNavigation, options]);
    const handleClick = useCallback((event, option) => {
        onSelect?.(option.value);
        if (option.href) {
            event.preventDefault();
            window.open(option.href, '_blank');
        }
        else {
            setSelectedValue(option.value);
        }
    }, [setSelectedValue, onSelect]);
    useEffect(() => {
        // If no option is highlighted, highlight the first one
        if (!highlightedValue && options.length > 0) {
            onHighlight(options[0].value);
        }
    }, [highlightedValue, onHighlight, options]);
    return (_jsx("div", { ref: listboxRef, role: "listbox", id: listboxId, className: className, tabIndex: 0, onKeyDown: handleKeyDown, "aria-activedescendant": highlightedValue ? `${listboxId}-${highlightedValue}` : undefined, css: css({
            outline: 'none',
            '&:focus-visible': {
                boxShadow: `0 0 0 2px ${theme.colors.actionDefaultBorderFocus}`,
                borderRadius: theme.borders.borderRadiusSm,
            },
        }), children: options.map((option) => (option.renderOption || ((additionalProps) => _jsx("div", { ...additionalProps, children: option.label })))({
            key: option.value,
            role: option.href ? 'link' : 'option',
            id: `${listboxId}-${option.value}`,
            'aria-selected': option.value === selectedValue,
            onClick: (event) => handleClick(event, option),
            onMouseEnter: () => onHighlight(option.value),
            'data-highlighted': option.value === highlightedValue,
            css: getComboboxOptionItemWrapperStyles(theme),
            href: option.href,
            tabIndex: -1,
        })) }));
};
//# sourceMappingURL=ListboxOptions.js.map