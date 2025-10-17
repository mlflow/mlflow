import { createContext, useContext, useMemo, useState, useCallback } from 'react';
import type { ReactNode } from 'react';

type HomePageViewState = {
  isLogTracesDrawerOpen: boolean;
  openLogTracesDrawer: () => void;
  closeLogTracesDrawer: () => void;
};

const HomePageViewStateContext = createContext<HomePageViewState>({
  isLogTracesDrawerOpen: false,
  openLogTracesDrawer: () => {},
  closeLogTracesDrawer: () => {},
});

export const HomePageViewStateProvider = ({ children }: { children: ReactNode }) => {
  const [isLogTracesDrawerOpen, setIsLogTracesDrawerOpen] = useState(false);

  const openLogTracesDrawer = useCallback(() => setIsLogTracesDrawerOpen(true), []);
  const closeLogTracesDrawer = useCallback(() => setIsLogTracesDrawerOpen(false), []);

  const value = useMemo(
    () => ({
      isLogTracesDrawerOpen,
      openLogTracesDrawer,
      closeLogTracesDrawer,
    }),
    [isLogTracesDrawerOpen, openLogTracesDrawer, closeLogTracesDrawer],
  );

  return <HomePageViewStateContext.Provider value={value}>{children}</HomePageViewStateContext.Provider>;
};

export const useHomePageViewState = () => useContext(HomePageViewStateContext);
