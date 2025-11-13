import { useCallback } from 'react';
/**
 * This hook provides a way to chain event handlers together.
 * Optionally, it can stop calling the next handler if the event has been defaultPrevented.
 * @param handlers Array of event handlers to chain together. Optional handlers are allowed for convenience.
 * @param stopOnDefaultPrevented If true, the next handler will not be called if the event has been defaultPrevented
 * @returns A function that will call each handler in the order they are provided
 * @example
 * ```tsx
 * const onClick = useChainEventHandlers({ handlers: [onClick1, onClick2] });
 * return <button onClick={onClick} />;
 */
export const useChainEventHandlers = (props) => {
    const { handlers, stopOnDefaultPrevented } = props;
    return useCallback((event) => {
        // Loop over each handler in succession
        for (const handler of handlers) {
            // Break if the event has been defaultPrevented and stopOnDefaultPrevented is true
            if (stopOnDefaultPrevented && event.defaultPrevented)
                return;
            // Call the handler if it exists
            handler?.(event);
        }
    }, [handlers, stopOnDefaultPrevented]);
};
//# sourceMappingURL=useChainEventHandlers.js.map