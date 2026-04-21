import { type CancellablePromise } from './fastdom';
type Task = () => void;
/**
 * Schedules fastdom tasks with automatic cleanup on unmount.
 * Propagates the first error in a batch to the nearest Error Boundary.
 */
export declare function useFastDom(): {
    measure: (fn: Task) => CancellablePromise;
    mutate: (fn: Task) => CancellablePromise;
};
export {};
//# sourceMappingURL=useFastDom.d.ts.map