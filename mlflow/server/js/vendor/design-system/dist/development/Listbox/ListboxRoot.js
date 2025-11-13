import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { createContext, useCallback, useContext, useMemo, useState } from 'react';
const ListboxContext = createContext(null);
export const useListboxContext = () => {
    const context = useContext(ListboxContext);
    if (!context) {
        throw new Error('useListboxContext must be used within a ListboxProvider');
    }
    return context;
};
export const ListboxRoot = ({ children, className, onSelect, initialSelectedValue, listBoxDivRef, }) => {
    const [selectedValue, setSelectedValue] = useState(initialSelectedValue);
    const [highlightedValue, setHighlightedValue] = useState();
    const listboxId = useMemo(() => `listbox-${Math.random().toString(36).slice(2)}`, []);
    const getContentOptions = (element) => {
        const options = element?.querySelectorAll('[role="option"], [role="link"]');
        return options ? Array.from(options) : undefined;
    };
    const handleKeyNavigation = useCallback((event, options) => {
        const currentIndex = options.findIndex((option) => option.value === highlightedValue);
        let nextIndex = currentIndex;
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                nextIndex = currentIndex < options.length - 1 ? currentIndex + 1 : 0;
                break;
            case 'ArrowUp':
                event.preventDefault();
                nextIndex = currentIndex > 0 ? currentIndex - 1 : options.length - 1;
                break;
            case 'Home':
                event.preventDefault();
                nextIndex = 0;
                break;
            case 'End':
                event.preventDefault();
                nextIndex = options.length - 1;
                break;
            case 'Enter':
            case ' ':
                event.preventDefault();
                if (highlightedValue !== undefined) {
                    onSelect?.(highlightedValue);
                    if (options[currentIndex].href) {
                        window.open(options[currentIndex].href, '_blank');
                    }
                    else {
                        setSelectedValue(highlightedValue);
                    }
                }
                break;
            default:
                return;
        }
        if (nextIndex !== currentIndex && listBoxDivRef?.current) {
            setHighlightedValue(options[nextIndex].value);
            const optionsList = getContentOptions(listBoxDivRef?.current);
            if (optionsList) {
                optionsList[nextIndex]?.scrollIntoView?.({ block: 'center' });
            }
        }
    }, [highlightedValue, onSelect, listBoxDivRef]);
    const contextValue = useMemo(() => ({
        selectedValue,
        setSelectedValue,
        highlightedValue,
        setHighlightedValue,
        listboxId,
        handleKeyNavigation,
    }), [selectedValue, highlightedValue, listboxId, handleKeyNavigation]);
    return (_jsx(ListboxContext.Provider, { value: contextValue, children: _jsx("div", { className: className, children: children }) }));
};
//# sourceMappingURL=ListboxRoot.js.map