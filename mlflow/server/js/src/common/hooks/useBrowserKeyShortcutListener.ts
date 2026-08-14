import { useEffect } from 'react';

declare interface NavigatorWithUserData extends Navigator {
  userAgentData: any;
}

const isMacKeyboard = () =>
  // userAgentData should be supported in modern Chromium based browsers
  /mac/i.test((window.navigator as NavigatorWithUserData).userAgentData?.platform) ||
  // if not, falls back to navigator.platform
  /mac/i.test(window.navigator.platform);

const systemModifierKey: keyof KeyboardEvent = isMacKeyboard() ? 'metaKey' : 'ctrlKey';

/**
 * Triggers certain action when a keyboard combination is pressed
 *
 * @example
 *
 * // Listens to CMD+S action
 * useBrowserKeyShortcutListener('s', { ctrlOrCmdKey: true }, () => { ... })
 */
export const useBrowserKeyShortcutListener = (
  /**
   * A single key (e.g. "s") that will be listened for pressing
   */
  key: string,
  /**
   * Determines which modifier keys are necessary to trigger the action
   */
  modifierKeys: { shiftKey?: boolean; ctrlOrCmdKey?: boolean; altOrOptKey?: boolean } = {},
  /**
   * A callback function. If returns true, the default action for the key combination will be prevented.
   */
  fn: () => boolean | void,
) => {
  const { altOrOptKey = false, ctrlOrCmdKey = false, shiftKey = false } = modifierKeys;
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        (!ctrlOrCmdKey || e[systemModifierKey]) &&
        (!altOrOptKey || e.altKey) &&
        (!shiftKey || e.shiftKey) &&
        e.key === key
      ) {
        const shouldPreventDefault = fn();
        if (shouldPreventDefault) {
          e.preventDefault();
        }
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [key, fn, ctrlOrCmdKey, altOrOptKey, shiftKey]);

  return { isMacKeyboard };
};
