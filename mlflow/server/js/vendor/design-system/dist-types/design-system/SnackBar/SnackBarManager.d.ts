export type SnackbarVertical = 'top' | 'bottom';
export type SnackbarHorizontal = 'left' | 'center' | 'right';
export interface SnackbarConfig {
    content: React.ReactNode;
    uid?: string;
    backgroundColor?: string;
    textColor?: string;
    vertical?: SnackbarVertical;
    horizontal?: SnackbarHorizontal;
    autoDismiss?: number;
    componentId?: string;
    closeButton?: boolean;
}
export interface SnackbarWithId extends SnackbarConfig {
    id: string;
    open: boolean;
}
type Listener = () => void;
/**
 * SnackBarManager is a store for managing SnackBar state.
 * It maintains the list of active snacks and notifies subscribers when state changes.
 */
declare class SnackBarManager {
    private snacks;
    private listeners;
    subscribe: (listener: Listener) => () => void;
    getSnapshot: () => SnackbarWithId[];
    private notifyListeners;
    addSnack(config: SnackbarConfig): string;
    closeSnack(id: string): void;
    deleteSnack(id: string): void;
    doesSnackExist(id: string): boolean;
    reset(): void;
}
export declare const snackBarManager: SnackBarManager;
export {};
//# sourceMappingURL=SnackBarManager.d.ts.map