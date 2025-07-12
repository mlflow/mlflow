import { useEffect, useRef } from 'react';
// Copied from https://usehooks-typescript.com/react-hook/use-interval
function useInterval(callback, delay) {
    const savedCallback = useRef(callback);
    // Remember the latest callback if it changes.
    useEffect(() => {
        savedCallback.current = callback;
    }, [callback]);
    // Set up the interval.
    useEffect(() => {
        // Don't schedule if no delay is specified.
        if (delay === null) {
            return;
        }
        const id = setInterval(() => savedCallback.current(), delay);
        return () => clearInterval(id);
    }, [delay]);
}
export default useInterval;
//# sourceMappingURL=useInterval.js.map