/// <reference types="react" />
interface useNotifyOnFirstViewProps {
    onView: () => void;
}
/**
 * Checks if the element was viewed and calls the onView callback.
 * NOTE: This hook only triggers the onView callback once for the element.
 * @param onView - callback to be called when the element is viewed
 * @typeParam T - extends Element to specify the type of element being observed
 */
export declare const useNotifyOnFirstView: <T extends Element>({ onView }: useNotifyOnFirstViewProps) => {
    elementRef: import("react").MutableRefObject<T | null>;
};
export {};
//# sourceMappingURL=useNotifyOnFirstView.d.ts.map