/**
 * A React custom hook that allows a callback function to be executed exactly once until it is explicitly reset.
 *
 * Usage:
 *
 * const originalCallback = () => { console.log('originalCallback'); }
 * const { callbackOnceUntilReset, reset } = useCallbackOnceUntilReset(originalCallback);
 *
 * // To execute the callback
 * callbackOnceUntilReset(); // Prints 'originalCallback'
 * callbackOnceUntilReset(); // No effect for further calls
 * reset();
 * callbackOnceUntilReset(); // Prints 'originalCallback' again
 */
declare const useCallbackOnceUntilReset: <T>(callback: () => T) => {
    callbackOnceUntilReset: () => void;
    reset: () => void;
};
export { useCallbackOnceUntilReset };
//# sourceMappingURL=useCallbackOnceUntilReset.d.ts.map