import type { ReactWrapper } from 'enzyme';
interface QueryOptions {
    /** Whether to trim the whitespace from DOM text using `String.prototype.trim` */
    trim?: boolean;
}
/**
 * Finds a single element that contains the specified text in the wrapper. If
 * there are 0 or more than 1 element that contains the specified text, an error
 * is thrown. Returns the element in an enzyme wrapper.
 */
export declare function findByText<P, S, C>(wrapper: ReactWrapper<P, S, C>, text: string | RegExp, queryOptions?: QueryOptions): ReactWrapper<any, any, import("react").Component<{}, {}, any>>;
/**
 * Finds all elements that contain the specified text. To avoid duplicate results,
 * only the parents of text nodes are returned.
 */
export declare function findAllByText<P, S, C>(wrapper: ReactWrapper<P, S, C>, text: string | RegExp, { trim }?: QueryOptions): ReactWrapper<any, any, import("react").Component<{}, {}, any>>[];
interface WaitForOptions {
    interval?: number;
    stackTraceError?: Error;
    timeout?: number;
}
/**
 * Wraps `_waitFor` in React's `act` testing utility. Used when the React component
 * updates during the execution of the callback (either because of indirect effects
 * being run or because of direct requests to update the component, like wrapper.update).
 * Prevents updates related to the callback from being affected by other updates
 * and more closely mimics how React runs in the browser. See
 * https://reactjs.org/docs/test-utils.html#act for more info on `act`.
 */
export declare function waitFor<T>(callback: () => T | Promise<T>, options?: WaitForOptions): Promise<T>;
/**
 * Finds all elements (that are rendered in the DOM) in `wrapper` that have an explicit
 * role of `role` specified. This is similar to `getAllByRole` from @testing-library/react
 * but is much simpler because of the shortcomings of Enzyme's API.
 */
export declare function findAllByRole<P, S, C>(wrapper: ReactWrapper<P, S, C>, role: string): ReactWrapper[];
/**
 * Finds a single element that has the specified role in the wrapper. If
 * there are 0 or more than 1 element that have that role, an error
 * is thrown. Returns the element in an enzyme wrapper.
 */
export declare function findByRole<P, S, C>(wrapper: ReactWrapper<P, S, C>, role: string): ReactWrapper;
export {};
//# sourceMappingURL=utils.d.ts.map