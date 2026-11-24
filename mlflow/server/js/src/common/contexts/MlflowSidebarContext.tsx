import { createContext, useContext } from 'react';

/**
 * Exposes the app-shell sidebar's collapse state (the `MlflowSidebar` next to the
 * MLflow logo) to deep pages so they can temporarily collapse it and then restore
 * whatever state it was in before.
 *
 * Why this exists: the review-queue focused-review view wants the full page width
 * while reviewing a trace, so it collapses the app-shell sidebar on entering focus
 * mode and restores the prior state on exit (a sidebar the user had already
 * collapsed must stay collapsed). The collapse state is owned by `MlflowRootLayout`
 * as local state with no other way to reach it from a routed page, hence this
 * context rather than prop-drilling through every page.
 */
export interface MlflowSidebarContextValue {
  showSidebar: boolean;
  setShowSidebar: (showSidebar: boolean) => void;
}

export const MlflowSidebarContext = createContext<MlflowSidebarContextValue | undefined>(undefined);

/** The app-shell sidebar toggle, or `undefined` when rendered outside the provider. */
export const useMlflowSidebar = (): MlflowSidebarContextValue | undefined => useContext(MlflowSidebarContext);
