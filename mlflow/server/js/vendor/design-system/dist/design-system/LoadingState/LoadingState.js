import { createContext, useContext, useLayoutEffect } from 'react';
import { useStableUid } from '../utils/useStableUid';
export const LoadingStateContext = createContext(null);
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
export const LoadingState = ({ description = 'Generic UI loading state', }) => {
    const uid = useStableUid();
    const loadingStateContext = useContext(LoadingStateContext);
    useLayoutEffect(() => {
        // mount
        if (loadingStateContext) {
            loadingStateContext.startLoading(uid, description);
        }
        return () => {
            // unmount
            if (loadingStateContext) {
                loadingStateContext.endLoading(uid);
            }
        };
    }, [uid, description, loadingStateContext]);
    return null;
};
//# sourceMappingURL=LoadingState.js.map