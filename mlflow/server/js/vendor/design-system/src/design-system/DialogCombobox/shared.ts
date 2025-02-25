import type React from 'react';

import type { Theme } from '../../theme';

export const getDialogComboboxOptionLabelWidth = (theme: Theme, width: number | string): number | string => {
  const paddingLeft = theme.spacing.xs + theme.spacing.sm;
  const iconWidth = theme.spacing.md;
  const labelMarginLeft = theme.spacing.sm;

  if (typeof width === 'string') {
    return `calc(${width} - ${paddingLeft + iconWidth + labelMarginLeft}px)`;
  }

  return width - paddingLeft + iconWidth + labelMarginLeft;
};

export function isOptionDisabled(option: HTMLElement): boolean {
  return option.hasAttribute('disabled') && option.getAttribute('disabled') !== 'false';
}

export function highlightFirstNonDisabledOption(
  firstOptionItem: Element,
  startAt: 'start' | 'end' = 'start',
  previousSelection?: HTMLElement,
): void {
  if (isOptionDisabled(firstOptionItem as HTMLElement)) {
    const firstHighlightableOption = findClosestOptionSibling(
      firstOptionItem as HTMLElement,
      startAt === 'end' ? 'previous' : 'next',
    );
    if (firstHighlightableOption) {
      highlightOption(firstHighlightableOption, previousSelection);
    }
  } else {
    highlightOption(firstOptionItem as HTMLElement, previousSelection);
  }
}

export function findClosestOptionSibling(element: HTMLElement, direction: 'previous' | 'next'): HTMLElement | null {
  const nextSibling = (
    direction === 'previous' ? element.previousElementSibling : element.nextElementSibling
  ) as HTMLElement;

  if (nextSibling?.getAttribute('role') === 'option') {
    if (isOptionDisabled(nextSibling)) {
      return findClosestOptionSibling(nextSibling, direction);
    }
    return nextSibling;
  } else if (nextSibling) {
    let nextOptionSibling = nextSibling;
    while (
      nextOptionSibling &&
      (nextOptionSibling.getAttribute('role') !== 'option' || isOptionDisabled(nextOptionSibling))
    ) {
      nextOptionSibling = (
        direction === 'previous' ? nextOptionSibling.previousElementSibling : nextOptionSibling.nextElementSibling
      ) as HTMLElement;
    }
    return nextOptionSibling;
  }
  return null;
}

const resetAllHighlightedOptions = (currentSelection: HTMLElement): void => {
  const options = getContentOptions(currentSelection);
  options?.forEach((option) => {
    option.setAttribute('tabIndex', '-1');
    option.setAttribute('data-highlighted', 'false');
  });
};

export const highlightOption = (currentSelection: HTMLElement, prevSelection?: HTMLElement, focus = true): void => {
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

export const findHighlightedOption = (options: HTMLElement[]): HTMLElement | undefined => {
  return options.find((option) => option.getAttribute('data-highlighted') === 'true') ?? undefined;
};

export const getContentOptions = (element: HTMLElement): HTMLElement[] | undefined => {
  const options = element.closest('[data-combobox-option-list="true"]')?.querySelectorAll('[role="option"]');
  return options ? (Array.from(options) as HTMLElement[]) : undefined;
};

export const getKeyboardNavigationFunctions = (
  handleSelect: (...args: any[]) => any,
  {
    onKeyDown,
    onMouseEnter,
    onDefaultKeyDown,
  }: {
    onKeyDown?: (...args: any[]) => any;
    onMouseEnter?: (...args: any[]) => any;
    onDefaultKeyDown?: (...args: any[]) => any;
  },
) => ({
  onKeyDown: (e: React.KeyboardEvent<HTMLDivElement>) => {
    onKeyDown?.(e);
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        const nextSibling = findClosestOptionSibling(e.currentTarget, 'next');
        if (nextSibling) {
          highlightOption(nextSibling, e.currentTarget);
        } else {
          const firstOption = getContentOptions(e.currentTarget)?.[0];
          if (firstOption) {
            highlightFirstNonDisabledOption(firstOption, 'start', e.currentTarget);
          }
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        const previousSibling = findClosestOptionSibling(e.currentTarget, 'previous');
        if (previousSibling) {
          highlightOption(previousSibling, e.currentTarget);
        } else {
          const lastOption = getContentOptions(e.currentTarget)?.slice(-1)[0];
          if (lastOption) {
            highlightFirstNonDisabledOption(lastOption as HTMLElement, 'end', e.currentTarget);
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
  onMouseEnter: (e: React.MouseEvent<HTMLDivElement>) => {
    onMouseEnter?.(e);
    resetTabIndexToFocusedElement(e.currentTarget);
  },
});

export const resetTabIndexToFocusedElement = (elem: HTMLElement) => {
  resetAllHighlightedOptions(elem);
  elem.setAttribute('tabIndex', '0');
  elem.focus();
};

export const dialogComboboxLookAheadKeyDown = (
  e: React.KeyboardEvent<any>,
  setLookAhead: (val: string) => void,
  lookAhead: string,
) => {
  if (e.key === 'Escape' || e.key === 'Tab' || e.key === 'Enter') {
    return;
  }
  e.preventDefault();
  const siblings = Array.from<HTMLElement>(e.currentTarget.parentElement?.children ?? []);
  // Look for the first sibling that starts with the pressed key + recently pressed keys (lookAhead, cleared after 1.5 seconds of inactivity)
  const nextSiblingIndex = siblings.findIndex((sibling) => {
    const siblingLabel = sibling.textContent?.toLowerCase() ?? '';
    return siblingLabel.startsWith(lookAhead + e.key);
  });

  if (nextSiblingIndex !== -1) {
    const nextSibling = siblings[nextSiblingIndex] as HTMLElement;
    nextSibling.focus();
    if (setLookAhead) {
      setLookAhead(lookAhead + e.key);
    }
    resetTabIndexToFocusedElement(nextSibling);
  }
};
