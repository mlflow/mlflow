type UseMediaQueryOptions = {
    defaultValue?: boolean;
    initializeWithValue?: boolean;
};
interface useMediaQueryProps {
    /** The media query to track. */
    query: string;
    /**
     * The default value to return if the hook is being run on the server (default is `false`).
     * options.defaultValue: The default value to return if the hook is being run on the server (default is `false`).
     * options.initializeWithValue: If `true` (default), the hook will initialize reading the media query. In SSR, you should set it to `false`, returning `options.defaultValue` or `false` initially.
     */
    options?: boolean | UseMediaQueryOptions;
}
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
export declare function useMediaQuery({ query, options }: useMediaQueryProps): boolean;
export {};
//# sourceMappingURL=useMediaQuery.d.ts.map