import { useState } from 'react';

/**
 * Manages refresh of charts. A simple publish-subscribe pattern that registers
 * callbacks and triggers them on refresh.
 */
export class ChartRefreshManager {
  private refreshCallbacks: (() => void)[] = [];
  public registerRefreshCallback(cb: () => void) {
    this.refreshCallbacks.push(cb);
    return () => {
      this.refreshCallbacks = this.refreshCallbacks.filter((c) => c !== cb);
    };
  }
  public refreshAllCharts() {
    this.refreshCallbacks.forEach((cb) => cb());
  }
}

/**
 * Returns local instance of chart refresh manager.
 */
export const useChartRefreshManager = () => {
  const [refreshManager] = useState(() => new ChartRefreshManager());
  return refreshManager;
};
