/**
 * A minimal DOM read/write scheduler to prevent layout thrashing.
 * Loosely based on https://github.com/wilsonpage/fastdom and using the same API.
 */
export type InstrumentedGlobal = typeof globalThis & {
    __perfHooks?: {
        rit?: {
            blockActiveInteractions?: (holdName: string) => () => void;
        };
    };
};
type TaskFn = () => void;
export interface CancellablePromise extends Promise<void> {
    cancel: () => void;
}
/**
 * Schedules a DOM read task (e.g., getBoundingClientRect, offsetWidth).
 * @returns Promise that resolves when task completes. Call .cancel() to abort.
 */
export declare function measure(fn: TaskFn): CancellablePromise;
/**
 * Schedules a DOM write task (e.g., style.width, classList.add).
 * @returns Promise that resolves when task completes. Call .cancel() to abort.
 */
export declare function update(fn: TaskFn): CancellablePromise;
/**
 * Flush all pending measure/update tasks immediately.
 * Intended for use in tests to avoid depending on requestAnimationFrame timing.
 */
export declare function flushFastdomForTesting(): void;
export {};
//# sourceMappingURL=fastdom.d.ts.map