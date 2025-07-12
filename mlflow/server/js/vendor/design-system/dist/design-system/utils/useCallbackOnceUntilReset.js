import { useCallback, useRef } from 'react';
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
const useCallbackOnceUntilReset = (callback) => {
    const canTriggerRef = useRef(true);
    const reset = useCallback(() => {
        canTriggerRef.current = true;
    }, []);
    const callbackOnceUntilReset = useCallback(() => {
        if (canTriggerRef.current) {
            callback();
            canTriggerRef.current = false;
        }
    }, [callback]);
    return { callbackOnceUntilReset, reset };
};
export { useCallbackOnceUntilReset };
//# sourceMappingURL=useCallbackOnceUntilReset.js.map