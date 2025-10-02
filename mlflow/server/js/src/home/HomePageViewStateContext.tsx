import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

type HomePageViewState = {
  // Tracing tutorial drawer
  isLogTracesDrawerOpen: boolean;
  openLogTracesDrawer: () => void;
  closeLogTracesDrawer: () => void;
  // Evaluation tutorial drawer
  isRunEvaluationDrawerOpen: boolean;
  openRunEvaluationDrawer: () => void;
  closeRunEvaluationDrawer: () => void;
};

const HomePageViewStateContext = createContext<HomePageViewState>({
  isLogTracesDrawerOpen: false,
  openLogTracesDrawer: () => {},
  closeLogTracesDrawer: () => {},
  isRunEvaluationDrawerOpen: false,
  openRunEvaluationDrawer: () => {},
  closeRunEvaluationDrawer: () => {},
});

export const HomePageViewStateProvider = ({ children }: { children: ReactNode }) => {
  const [isLogTracesDrawerOpen, setIsLogTracesDrawerOpen] = useState(false);
  const [isRunEvaluationDrawerOpen, setIsRunEvaluationDrawerOpen] = useState(false);

  const openLogTracesDrawer = useCallback(() => setIsLogTracesDrawerOpen(true), []);
  const closeLogTracesDrawer = useCallback(() => setIsLogTracesDrawerOpen(false), []);

  const openRunEvaluationDrawer = useCallback(() => setIsRunEvaluationDrawerOpen(true), []);
  const closeRunEvaluationDrawer = useCallback(() => setIsRunEvaluationDrawerOpen(false), []);

  const value = useMemo(
    () => ({
      isLogTracesDrawerOpen,
      openLogTracesDrawer,
      closeLogTracesDrawer,
      isRunEvaluationDrawerOpen,
      openRunEvaluationDrawer,
      closeRunEvaluationDrawer,
    }),
    [
      isLogTracesDrawerOpen,
      openLogTracesDrawer,
      closeLogTracesDrawer,
      isRunEvaluationDrawerOpen,
      openRunEvaluationDrawer,
      closeRunEvaluationDrawer,
    ],
  );

  return <HomePageViewStateContext.Provider value={value}>{children}</HomePageViewStateContext.Provider>;
};

export const useHomePageViewState = () => useContext(HomePageViewStateContext);
