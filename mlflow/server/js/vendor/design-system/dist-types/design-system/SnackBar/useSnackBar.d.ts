import type { SnackbarConfig } from './SnackBarManager';
/**
 * Hook to interact with the SnackBar system.
 * Use this hook to add and close snackbars.
 */
export declare function useSnackBar(): {
    addSnack: (config: SnackbarConfig) => string;
    closeSnack: (id: string) => void;
};
//# sourceMappingURL=useSnackBar.d.ts.map