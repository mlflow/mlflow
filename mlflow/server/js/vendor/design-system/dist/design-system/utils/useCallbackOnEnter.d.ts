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
export declare const useCallbackOnEnter: <T extends Element>({ callback, allowBasicEnter, allowPlatformEnter, }: {
    callback: React.KeyboardEventHandler<T>;
    allowBasicEnter: boolean;
    allowPlatformEnter: boolean;
}) => {
    onKeyDown: import("react").KeyboardEventHandler<T>;
    onCompositionEnd: import("react").CompositionEventHandler<T>;
    onCompositionStart: import("react").CompositionEventHandler<T>;
};
//# sourceMappingURL=useCallbackOnEnter.d.ts.map