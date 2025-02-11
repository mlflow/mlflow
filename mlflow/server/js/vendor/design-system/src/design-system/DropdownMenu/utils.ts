const getAllMenuItemsInContainer = (container: Element) => {
  return container.querySelectorAll('[role^="menuitem"]');
};

const focusNextItem = (e: React.KeyboardEvent) => {
  const container = e.currentTarget.closest('[role="menu"]');
  if (!container) {
    return;
  }

  const menuItems = getAllMenuItemsInContainer(container);
  const activeElement = document.activeElement as HTMLElement;
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
        (tooltip as HTMLElement).focus();
      }
    } else {
      (nextItem as HTMLElement).focus();
      (nextItem as HTMLElement).setAttribute('data-highlighted', 'true');
    }
  }
};

export const blurTooltipAndFocusNextItem = (e: React.KeyboardEvent) => {
  const tooltip = document.activeElement as HTMLElement;
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
        (tooltip as HTMLElement).focus();
      }
    } else {
      (nextItem as HTMLElement).focus();
    }
  }
};

export const handleKeyboardNavigation = (e: React.KeyboardEvent) => {
  const isItemFocused =
    document.activeElement?.getAttribute('role') === 'menuitem' ||
    document.activeElement?.getAttribute('role') === 'menuitemcheckbox' ||
    document.activeElement?.getAttribute('role') === 'menuitemradio';
  const isTooltipFocused = document.activeElement?.hasAttribute('data-disabled-tooltip');
  if (isItemFocused || !isTooltipFocused) {
    focusNextItem(e);
  } else {
    blurTooltipAndFocusNextItem(e);
  }
};
