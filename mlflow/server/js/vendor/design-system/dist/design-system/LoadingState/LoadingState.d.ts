/** Props for UI components that support a "loading" state. */
export interface WithLoadingState {
    /**
     * Whether the component is in a "loading" state, i.e. the user has an expectation
     * that the UI is not fully rendered and is still waiting on some data to become
     * available.
     */
    loading?: boolean;
    /**
     * An optional description of the semantic meaning of the UI component in a
     * "loading" state, used for logging and debugging. For example, when a `<Spinner>`
     * is used within a side panel, you could specify something like "Query plan panel"
     * to make it clear what the spinner represents.
     */
    loadingDescription?: string;
}
/**
 * A handler for integrating UI components with external latency instrumentation.
 * If provided via `LoadingStateContext`, hooks will be called whenever child
 * components in a "loading" state are mounted and unmounted.
 */
export interface LoadingStateHandler {
    startLoading(uid: number, description: string): void;
    endLoading(uid: number): void;
}
export declare const LoadingStateContext: import("react").Context<LoadingStateHandler | null>;
type LoadingStateProps = {
    description?: string;
};
/**
 * Indicates that the containing component is in a "loading" state, i.e. that the UI
 * displayed to the user is semantically equivalent to them seeing a spinner or a
 * loading indicator. This means that the UI is not in its final settled state yet.
 *
 * All components that are in a "loading" state should render a `<LoadingState>`
 * component, preferrably with an appropriate description.
 *
 * By itself, `<LoadingState>` doesn't do anything, but if used within `LoadingStateContext`,
 * it will call the provided `startLoading()`/`endLoading()` hooks when the component
 * is mounted/unmounted, which can be used to integrate existing latency instrumentation.
 */
export declare const LoadingState: React.FC<LoadingStateProps>;
export {};
//# sourceMappingURL=LoadingState.d.ts.map