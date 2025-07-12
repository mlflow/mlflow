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
export declare const useChainEventHandlers: <T extends React.SyntheticEvent<any>>(props: {
    handlers: Array<React.EventHandler<T> | undefined>;
    stopOnDefaultPrevented?: boolean;
}) => (event: T) => void;
//# sourceMappingURL=useChainEventHandlers.d.ts.map