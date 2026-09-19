import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { MutableRefObject, ReactNode } from 'react';

// Override/Discard handlers for the currently-applied shared view. Held in a ref (see below) rather
// than state so republishing them as their identity changes never triggers a render.
interface SharedViewActionHandlers {
  override: () => void;
  discard: () => void;
}

interface SharedViewActionsBridgeValue {
  // Whether a shared view is currently applied. React state, so flipping it re-renders consumers
  // (the header Views menu, which shows Override/Discard only when a view is active).
  active: boolean;
  setActive: (active: boolean) => void;
  // Latest override/discard handlers. A ref (not state) so ExperimentView can refresh them every
  // render without forcing a parent re-render — which, combined with the handlers' changing
  // identity, would loop. Read at click time.
  handlersRef: MutableRefObject<SharedViewActionHandlers | null>;
}

const SharedViewActionsBridgeContext = createContext<SharedViewActionsBridgeValue | null>(null);

/**
 * Bridges the runs shared-view Override/Discard actions from `ExperimentView` (which owns
 * `sharedViewActive` and the handlers) up to the header Views dropdown, which lives in a HIGHER tree
 * (`ExperimentPageTabs`) via the router outlet and so can't read that state directly.
 *
 * Safe as a single shared slot because there is exactly one `ExperimentView` per page (no modal
 * mounts a second one), so only one publisher ever writes here.
 */
export const SharedViewActionsBridgeProvider = ({ children }: { children: ReactNode }) => {
  const [active, setActive] = useState(false);
  const handlersRef = useRef<SharedViewActionHandlers | null>(null);
  const value = useMemo(() => ({ active, setActive, handlersRef }), [active]);
  return <SharedViewActionsBridgeContext.Provider value={value}>{children}</SharedViewActionsBridgeContext.Provider>;
};

/**
 * Called by `ExperimentView` to publish the current shared-view state/handlers upward. No-op when
 * rendered outside the provider (e.g. the standalone/legacy experiment page), so those surfaces are
 * unaffected.
 */
export const usePublishSharedViewActions = ({
  active,
  override,
  discard,
}: {
  active: boolean;
  override: () => void;
  discard: () => void;
}) => {
  const ctx = useContext(SharedViewActionsBridgeContext);

  // Keep the latest handlers in the shared ref. Writing a ref during render is the standard
  // latest-value pattern; it never triggers a re-render.
  if (ctx) {
    ctx.handlersRef.current = { override, discard };
  }

  const setActive = ctx?.setActive;
  useEffect(() => {
    if (!setActive) {
      return;
    }
    setActive(active);
    return () => setActive(false);
  }, [setActive, active]);
};

/**
 * Read the bridged shared-view actions from the header Views button. Returns null when no shared
 * view is applied (or outside the provider), so the menu shows no Override/Discard entries. The
 * returned handlers read the latest closures from the ref at call time.
 */
export const useSharedViewActionsBridge = (): SharedViewActionHandlers | null => {
  const ctx = useContext(SharedViewActionsBridgeContext);
  if (!ctx?.active) {
    return null;
  }
  return {
    override: () => ctx.handlersRef.current?.override(),
    discard: () => ctx.handlersRef.current?.discard(),
  };
};
