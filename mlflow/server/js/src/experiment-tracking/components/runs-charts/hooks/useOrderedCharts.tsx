import { first, indexOf, last, orderBy } from 'lodash';
import { useMemo, useRef, useState } from 'react';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';

const STORAGE_ITEM_KEY = 'ChartOrder';

/**
 * Provides convenience functions for move chart up/down actions
 */
export const useChartMoveUpDownFunctions = (
  metricKeys: string[],
  onReorder: (fromElement: string, toElement: string) => void,
) => {
  const canMoveUp = (metricKey: string) => metricKey !== first(metricKeys);
  const canMoveDown = (metricKey: string) => metricKey !== last(metricKeys);
  const moveChartUp = (metricKey: string) => {
    const previous = metricKeys[indexOf(metricKeys, metricKey) - 1];
    onReorder(metricKey, previous);
  };
  const moveChartDown = (metricKey: string) => {
    const next = metricKeys[indexOf(metricKeys, metricKey) + 1];
    onReorder(metricKey, next);
  };

  return { canMoveDown, canMoveUp, moveChartDown, moveChartUp };
};

export const useOrderedCharts = (metricKeys: string[], storageNamespace: string, storageKey: string) => {
  const localStorageStore = useRef(LocalStorageUtils.getStoreForComponent(storageNamespace, storageKey));

  const [currentOrder, setCurrentOrder] = useState<string[] | undefined>(() => {
    try {
      const savedOrder = localStorageStore.current.getItem(STORAGE_ITEM_KEY);
      if (savedOrder) return JSON.parse(savedOrder);
    } catch {
      return undefined;
    }
    return undefined;
  });

  const onReorderChart = (
    /**
     * "From" chart key, e.g. idenfifier of chart being dragged
     */
    fromMetricKey: string,
    /**
     * "To" chart key, e.g. idenfifier of chart being dropped on
     */
    toMetricKey: string,
  ) => {
    const sourceOrder = currentOrder ? [...currentOrder] : [...metricKeys];
    const indexSource = indexOf(sourceOrder, fromMetricKey);
    const indexTarget = indexOf(sourceOrder, toMetricKey);

    // Swap chart places
    [sourceOrder[indexSource], sourceOrder[indexTarget]] = [sourceOrder[indexTarget], sourceOrder[indexSource]];

    // Save the new order and persist it
    setCurrentOrder(sourceOrder);
    localStorageStore.current.setItem(STORAGE_ITEM_KEY, JSON.stringify(sourceOrder));
  };

  const orderedMetricKeys = useMemo(
    () =>
      orderBy(metricKeys, (key) =>
        currentOrder?.includes(key) ? indexOf(currentOrder, key) : Number.MAX_SAFE_INTEGER,
      ),
    [currentOrder, metricKeys],
  );

  return { orderedMetricKeys, onReorderChart };
};
