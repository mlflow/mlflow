const getAllMenuItemsInContainer = (container) => {
    return container.querySelectorAll('[role^="menuitem"]');
};
const focusNextItem = (e) => {
    const container = e.currentTarget.closest('[role="menu"]');
    if (!container) {
        return;
    }
    const menuItems = getAllMenuItemsInContainer(container);
    const activeElement = document.activeElement;
    const shouldNavigateUp = e.key === 'ArrowUp' || (e.key === 'Tab' && e.shiftKey);
    const activeIndex = Array.from(menuItems).findIndex((item) => item === activeElement);
    let nextIndex = shouldNavigateUp ? activeIndex - 1 : activeIndex + 1;
    if (nextIndex < 0 || nextIndex >= menuItems.length) {
        nextIndex = shouldNavigateUp ? menuItems.length - 1 : 0;
    }
    const nextItem = menuItems[nextIndex];
    if (nextItem) {
        const isDisabled = nextItem.hasAttribute('data-disabled');
        if (isDisabled) {
            const tooltip = nextItem.querySelector('[data-disabled-tooltip]');
            tooltip?.setAttribute('tabindex', '0');
            if (tooltip) {
                e.preventDefault();
                tooltip.focus();
            }
        }
        else {
            nextItem.focus();
            nextItem.setAttribute('data-highlighted', 'true');
        }
    }
};
export const blurTooltipAndFocusNextItem = (e) => {
    const tooltip = document.activeElement;
    const parentItem = tooltip.closest('[role^="menuitem"]');
    const container = tooltip.closest('[role="menu"]');
    if (!container) {
        return;
    }
    const menuItems = getAllMenuItemsInContainer(container);
    const activeIndex = Array.from(menuItems).findIndex((item) => item === parentItem);
    const shouldNavigateUp = e.key === 'ArrowUp' || (e.key === 'Tab' && e.shiftKey);
    let nextIndex = shouldNavigateUp ? activeIndex - 1 : activeIndex + 1;
    if (nextIndex < 0 || nextIndex >= menuItems.length) {
        nextIndex = shouldNavigateUp ? menuItems.length - 1 : 0;
    }
    const nextItem = menuItems[nextIndex];
    if (nextItem) {
        tooltip.removeAttribute('tabindex');
        tooltip.blur();
        const isDisabled = nextItem.hasAttribute('data-disabled');
        if (isDisabled) {
            const tooltip = nextItem.querySelector('[data-disabled-tooltip]');
            tooltip?.setAttribute('tabindex', '0');
            if (tooltip) {
                e.preventDefault();
                tooltip.focus();
            }
        }
        else {
            nextItem.focus();
        }
    }
};
export const handleKeyboardNavigation = (e) => {
    const isItemFocused = document.activeElement?.getAttribute('role') === 'menuitem' ||
        document.activeElement?.getAttribute('role') === 'menuitemcheckbox' ||
        document.activeElement?.getAttribute('role') === 'menuitemradio';
    const isTooltipFocused = document.activeElement?.hasAttribute('data-disabled-tooltip');
    if (isItemFocused || !isTooltipFocused) {
        focusNextItem(e);
    }
    else {
        blurTooltipAndFocusNextItem(e);
    }
};
//# sourceMappingURL=utils.js.map