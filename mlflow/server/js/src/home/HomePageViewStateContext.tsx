import { create } from '@databricks/web-shared/zustand';

type HomePageViewState = {
  isLogTracesDrawerOpen: boolean;
  openLogTracesDrawer: () => void;
  closeLogTracesDrawer: () => void;
};

export const useHomePageViewState = create<HomePageViewState>((set) => ({
  isLogTracesDrawerOpen: false,
  openLogTracesDrawer: () => set({ isLogTracesDrawerOpen: true }),
  closeLogTracesDrawer: () => set({ isLogTracesDrawerOpen: false }),
}));
