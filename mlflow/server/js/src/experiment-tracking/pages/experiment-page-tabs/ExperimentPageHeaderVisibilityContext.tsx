import { createContext, useContext, useMemo, useState } from 'react';

type HeaderVisibilityContextValue = {
  headerHidden: boolean;
  setHeaderHidden: (hidden: boolean) => void;
};

const HeaderVisibilityContext = createContext<HeaderVisibilityContextValue>({
  headerHidden: false,
  setHeaderHidden: () => {},
});

export const HeaderVisibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [headerHidden, setHeaderHidden] = useState(false);
  const value = useMemo(() => ({ headerHidden, setHeaderHidden }), [headerHidden]);
  return <HeaderVisibilityContext.Provider value={value}>{children}</HeaderVisibilityContext.Provider>;
};

export const useHeaderVisibility = () => useContext(HeaderVisibilityContext);
