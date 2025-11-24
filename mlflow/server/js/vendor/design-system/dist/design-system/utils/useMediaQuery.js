import { useState } from 'react';
import { useIsomorphicLayoutEffect } from './useIsomorphicLayoutEffect';
const IS_SERVER = typeof window === 'undefined';
/**
 * Copied from usehooks-ts.
 * Custom hook for tracking the state of a media query. Returns The current state of the media query (true if the query matches, false otherwise).
 *
 * [Documentation](https://usehooks-ts.com/react-hook/use-media-query)
 *
 * [MDN Match Media](https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia)
 *
 * Example:
 *
 * `const isSmallScreen = useMediaQuery('(max-width: 600px)');`
 *
 * Use `isSmallScreen` to conditionally apply styles or logic based on the screen size.
 */
export function useMediaQuery({ query, options }) {
    // TODO: Refactor this code after the deprecated signature has been removed.
    const defaultValue = typeof options === 'boolean' ? options : options?.defaultValue ?? false;
    const initializeWithValue = typeof options === 'boolean' ? undefined : options?.initializeWithValue ?? undefined;
    const [matches, setMatches] = useState(() => {
        if (initializeWithValue) {
            return getMatches(query);
        }
        return defaultValue;
    });
    const getMatches = (query) => {
        if (IS_SERVER) {
            return defaultValue;
        }
        return window.matchMedia(query).matches;
    };
    /** Handles the change event of the media query. */
    function handleChange() {
        setMatches(getMatches(query));
    }
    useIsomorphicLayoutEffect(() => {
        const matchMedia = window.matchMedia(query);
        // Triggered at the first client-side load and if query changes
        handleChange();
        // Use deprecated `addListener` and `removeListener` to support Safari < 14 (#135)
        if (matchMedia.addListener) {
            matchMedia.addListener(handleChange);
        }
        else {
            matchMedia.addEventListener('change', handleChange);
        }
        return () => {
            if (matchMedia.removeListener) {
                matchMedia.removeListener(handleChange);
            }
            else {
                matchMedia.removeEventListener('change', handleChange);
            }
        };
    }, [query]);
    return matches;
}
//# sourceMappingURL=useMediaQuery.js.map