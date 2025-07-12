export const getDialogComboboxOptionLabelWidth = (theme, width) => {
    const paddingLeft = theme.spacing.xs + theme.spacing.sm;
    const iconWidth = theme.spacing.md;
    const labelMarginLeft = theme.spacing.sm;
    if (typeof width === 'string') {
        return `calc(${width} - ${paddingLeft + iconWidth + labelMarginLeft}px)`;
    }
    return width - paddingLeft + iconWidth + labelMarginLeft;
};
export function isOptionDisabled(option) {
    return option.hasAttribute('disabled') && option.getAttribute('disabled') !== 'false';
}
export function highlightFirstNonDisabledOption(firstOptionItem, startAt = 'start', previousSelection) {
    if (isOptionDisabled(firstOptionItem)) {
        const firstHighlightableOption = findClosestOptionSibling(firstOptionItem, startAt === 'end' ? 'previous' : 'next');
        if (firstHighlightableOption) {
            highlightOption(firstHighlightableOption, previousSelection);
        }
    }
    else {
        highlightOption(firstOptionItem, previousSelection);
    }
}
export function findClosestOptionSibling(element, direction) {
    const nextSibling = (direction === 'previous' ? element.previousElementSibling : element.nextElementSibling);
    if (nextSibling?.getAttribute('role') === 'option') {
        if (isOptionDisabled(nextSibling)) {
            return findClosestOptionSibling(nextSibling, direction);
        }
        return nextSibling;
    }
    else if (nextSibling) {
        let nextOptionSibling = nextSibling;
        while (nextOptionSibling &&
            (nextOptionSibling.getAttribute('role') !== 'option' || isOptionDisabled(nextOptionSibling))) {
            nextOptionSibling = (direction === 'previous' ? nextOptionSibling.previousElementSibling : nextOptionSibling.nextElementSibling);
        }
        return nextOptionSibling;
    }
    return null;
}
const resetAllHighlightedOptions = (currentSelection) => {
    const options = getContentOptions(currentSelection);
    options?.forEach((option) => {
        option.setAttribute('tabIndex', '-1');
        option.setAttribute('data-highlighted', 'false');
    });
};
export const highlightOption = (currentSelection, prevSelection, focus = true) => {
    if (prevSelection) {
        prevSelection.setAttribute('tabIndex', '-1');
        prevSelection.setAttribute('data-highlighted', 'false');
    }
    if (focus) {
        currentSelection.focus();
    }
    currentSelection.setAttribute('tabIndex', '0');
    currentSelection.setAttribute('data-highlighted', 'true');
    currentSelection.scrollIntoView?.({ block: 'center' });
};
export const findHighlightedOption = (options) => {
    return options.find((option) => option.getAttribute('data-highlighted') === 'true') ?? undefined;
};
export const getContentOptions = (element) => {
    const options = element.closest('[data-combobox-option-list="true"]')?.querySelectorAll('[role="option"]');
    return options ? Array.from(options) : undefined;
};
export const getKeyboardNavigationFunctions = (handleSelect, { onKeyDown, onMouseEnter, onDefaultKeyDown, disableMouseOver, setDisableMouseOver, }) => ({
    onKeyDown: (e) => {
        onKeyDown?.(e);
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setDisableMouseOver(true);
                const nextSibling = findClosestOptionSibling(e.currentTarget, 'next');
                if (nextSibling) {
                    highlightOption(nextSibling, e.currentTarget);
                }
                else {
                    const firstOption = getContentOptions(e.currentTarget)?.[0];
                    if (firstOption) {
                        highlightFirstNonDisabledOption(firstOption, 'start', e.currentTarget);
                    }
                }
                break;
            case 'ArrowUp':
                e.preventDefault();
                setDisableMouseOver(true);
                const previousSibling = findClosestOptionSibling(e.currentTarget, 'previous');
                if (previousSibling) {
                    highlightOption(previousSibling, e.currentTarget);
                }
                else {
                    const lastOption = getContentOptions(e.currentTarget)?.slice(-1)[0];
                    if (lastOption) {
                        highlightFirstNonDisabledOption(lastOption, 'end', e.currentTarget);
                    }
                }
                break;
            case 'Enter':
                e.preventDefault();
                handleSelect(e);
                break;
            default:
                onDefaultKeyDown?.(e);
                break;
        }
    },
    onMouseMove: (e) => {
        if (disableMouseOver) {
            setDisableMouseOver(false);
        }
    },
    onMouseEnter: (e) => {
        if (!disableMouseOver) {
            onMouseEnter?.(e);
            resetTabIndexToFocusedElement(e.currentTarget);
        }
    },
});
export const resetTabIndexToFocusedElement = (elem) => {
    resetAllHighlightedOptions(elem);
    elem.setAttribute('tabIndex', '0');
    elem.focus();
};
export const dialogComboboxLookAheadKeyDown = (e, setLookAhead, lookAhead) => {
    if (e.key === 'Escape' || e.key === 'Tab' || e.key === 'Enter') {
        return;
    }
    e.preventDefault();
    const siblings = Array.from(e.currentTarget.parentElement?.children ?? []);
    // Look for the first sibling that starts with the pressed key + recently pressed keys (lookAhead, cleared after 1.5 seconds of inactivity)
    const nextSiblingIndex = siblings.findIndex((sibling) => {
        const siblingLabel = sibling.textContent?.toLowerCase() ?? '';
        return siblingLabel.startsWith(lookAhead + e.key);
    });
    if (nextSiblingIndex !== -1) {
        const nextSibling = siblings[nextSiblingIndex];
        nextSibling.focus();
        if (setLookAhead) {
            setLookAhead(lookAhead + e.key);
        }
        resetTabIndexToFocusedElement(nextSibling);
    }
};
//# sourceMappingURL=shared.js.map