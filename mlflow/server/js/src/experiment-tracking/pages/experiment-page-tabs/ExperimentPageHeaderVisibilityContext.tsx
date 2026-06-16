import { createContext, useContext, useMemo, useState } from 'react';

type HeaderVisibilityContextValue = {
  headerHidden: boolean;
  setHeaderHidden: (hidden: boolean) => void;
  /** When true, the header's action buttons (management menu, share, description edit)
   *  are hidden so the active tab page can provide its own controls without duplication. */
  headerActionsHidden: boolean;
  setHeaderActionsHidden: (hidden: boolean) => void;
};

const HeaderVisibilityContext = createContext<HeaderVisibilityContextValue>({
  headerHidden: false,
  setHeaderHidden: () => {},
  headerActionsHidden: false,
  setHeaderActionsHidden: () => {},
});

export const HeaderVisibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [headerHidden, setHeaderHidden] = useState(false);
  const [headerActionsHidden, setHeaderActionsHidden] = useState(false);
  const value = useMemo(
    () => ({ headerHidden, setHeaderHidden, headerActionsHidden, setHeaderActionsHidden }),
    [headerHidden, headerActionsHidden],
  );
  return <HeaderVisibilityContext.Provider value={value}>{children}</HeaderVisibilityContext.Provider>;
};

export const useHeaderVisibility = () => useContext(HeaderVisibilityContext);
