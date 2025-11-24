import { useCallback, useMemo, useRef } from 'react';
/**
 * ES-1267895 When entering composed characters, we should not submit forms on Enter.
 * For instance Japanese characters are composed and we should not submit forms when
 * the user is still composing the characters.
 *
 * This hook provides a reusable way to invoke a callback on Enter,
 * but not when composing characters.
 * This can be used to invoke a form submission callback when Enter is pressed.
 * @param callback VoidFunction to call when the Enter is pressed
 * @param allowBasicEnter If true, the callback will be invoked when Enter is pressed without any modifiers
 * @param allowPlatformEnter If true, the callback will be invoked when Enter is pressed with the platform modifier (CMD on Mac, CTRL on Windows)
 * @returns Object with onKeyDown, onCompositionEnd, and onCompositionStart event handlers
 *
 * @example
 * ```tsx
 * const handleSubmit = (event: React.KeyboardEvent) => {
 *  event.preventDefault();
 * // Submit the form
 * };
 * const eventHandlers = useCallbackOnEnter({
 *   callback: handleSubmit,
 *   allowBasicEnter: true,
 *   allowPlatformEnter: true,
 * })
 * return <input {...eventHandlers} />;
 * ```
 */
export const useCallbackOnEnter = ({ callback, allowBasicEnter, allowPlatformEnter, }) => {
    const isMacOs = useMemo(() => navigator.userAgent.includes('Mac'), []);
    // Keeping track of whether we are composing characters
    // This is stored in a ref so that it can be accessed in the onKeyDown event handler
    // without causing a re-renders
    const isComposing = useRef(false);
    // Handler for when the composition starts
    const onCompositionStart = useCallback(() => {
        isComposing.current = true;
    }, []);
    // Handler for when the composition ends
    const onCompositionEnd = useCallback(() => {
        isComposing.current = false;
    }, []);
    // Handler for when a key is pressed
    // Used to submit the form when Enter is pressed
    const onKeyDown = useCallback((event) => {
        // Only invoke the callback on Enter
        if (event.key !== 'Enter')
            return;
        // Do not submit on Enter if user is composing characters
        if (isComposing.current)
            return;
        // Do not submit on Enter if both are false
        if (!allowBasicEnter && !allowPlatformEnter)
            return;
        // Check if the event is a valid Enter press
        const basicEnter = allowBasicEnter && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey;
        const platformEnter = allowPlatformEnter && (isMacOs ? event.metaKey : event.ctrlKey);
        const isValidEnterPress = basicEnter || platformEnter;
        // Submit the form if the Enter press is valid
        if (isValidEnterPress)
            callback(event);
    }, [allowBasicEnter, allowPlatformEnter, callback, isMacOs]);
    return { onKeyDown, onCompositionEnd, onCompositionStart };
};
//# sourceMappingURL=useCallbackOnEnter.js.map