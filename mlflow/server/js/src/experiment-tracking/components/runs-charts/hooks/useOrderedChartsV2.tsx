import { first, last, orderBy } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import { RunViewMetricConfig } from '../../../types';

const STORAGE_ITEM_KEY = 'ChartOrder';
const SECTION_STORAGE_ITEM_KEY = 'SectionExpanded';

const includesKey = (sourceOrder: RunViewMetricConfig[], comparedMetricKey: string) =>
  sourceOrder.some(({ metricKey }) => metricKey === comparedMetricKey);

/**
 * Provides convenience functions for move chart up/down actions
 */
export const useChartMoveUpDownFunctionsV2 = (
  metricConfigs: RunViewMetricConfig[],
  onReorder: (fromElement: string, toElement: string) => void,
) => {
  const section2MetricConfigs: Record<string, RunViewMetricConfig[]> = {};
  metricConfigs.forEach((config: RunViewMetricConfig) => {
    if (section2MetricConfigs[config.sectionKey]) {
      section2MetricConfigs[config.sectionKey].push(config);
    } else {
      section2MetricConfigs[config.sectionKey] = [config];
    }
  });

  // Mapping of group to metricKeys
  const canMoveUp = (config: RunViewMetricConfig) => {
    const sectionMetricCharts = section2MetricConfigs[config.sectionKey];
    return first(sectionMetricCharts)?.metricKey !== config.metricKey;
  };
  const canMoveDown = (config: RunViewMetricConfig) => {
    const sectionMetricCharts = section2MetricConfigs[config.sectionKey];
    return last(sectionMetricCharts)?.metricKey !== config.metricKey;
  };
  const moveChartUp = (config: RunViewMetricConfig) => {
    const sectionMetricCharts = section2MetricConfigs[config.sectionKey];
    if (sectionMetricCharts) {
      const previous =
        sectionMetricCharts[sectionMetricCharts.findIndex((chart) => chart.metricKey === config.metricKey) - 1];
      onReorder(config.metricKey, previous.metricKey);
    }
  };
  const moveChartDown = (config: RunViewMetricConfig) => {
    const sectionMetricCharts = section2MetricConfigs[config.sectionKey];
    if (sectionMetricCharts) {
      const next =
        sectionMetricCharts[sectionMetricCharts.findIndex((chart) => chart.metricKey === config.metricKey) + 1];
      onReorder(config.metricKey, next.metricKey);
    }
  };

  return { canMoveDown, canMoveUp, moveChartDown, moveChartUp };
};

export const useOrderedChartsV2 = (
  metricConfigs: RunViewMetricConfig[],
  storageNamespace: string,
  storageKey: string,
) => {
  const localStorageStore = useRef(LocalStorageUtils.getStoreForComponent(storageNamespace, storageKey));

  const [currentOrder, setCurrentOrder] = useState<RunViewMetricConfig[] | undefined>(() => {
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
     * "From" chart key, e.g. identifier of chart being dragged
     */
    fromMetricKey: string,
    /**
     * "To" chart key, e.g. identifier of chart being dropped on
     */
    toMetricKey: string,
  ) => {
    const sourceOrder = currentOrder ? [...currentOrder] : [...metricConfigs];
    const indexSource = sourceOrder.findIndex((chart) => chart.metricKey === fromMetricKey);
    const indexTarget = sourceOrder.findIndex((chart) => chart.metricKey === toMetricKey);

    // Update metric chart section to new sectionId
    const fromMetric = sourceOrder[indexSource];
    const toMetric = sourceOrder[indexTarget];
    const isSameSection = fromMetric.sectionKey === toMetric.sectionKey;
    const newFromMetric = { ...fromMetric };
    newFromMetric.sectionKey = toMetric.sectionKey;
    // Insert chart at new position
    sourceOrder.splice(indexSource, 1);
    if (!isSameSection) {
      const newSectionIndex = sourceOrder.findIndex((chart) => chart.metricKey === toMetricKey);
      sourceOrder.splice(newSectionIndex, 0, newFromMetric);
    } else {
      sourceOrder.splice(indexTarget, 0, newFromMetric);
    }
    // Save the new order and persist it
    setCurrentOrder(sourceOrder);
    localStorageStore.current.setItem(STORAGE_ITEM_KEY, JSON.stringify(sourceOrder));
  };

  const onInsertChart = (
    /**
     * "From" chart key, e.g. identifier of chart being dragged
     */
    fromMetricKey: string,
    /**
     * "To" section group id, e.g. identifier of section being dropped on
     */
    toSectionId: string,
  ) => {
    const sourceOrder = currentOrder ? [...currentOrder] : [...metricConfigs];
    const indexSource = sourceOrder.findIndex((chart) => chart.metricKey === fromMetricKey);
    // Update group to panel
    const fromMetric = sourceOrder[indexSource];
    const newFromMetric = { ...fromMetric };
    newFromMetric.sectionKey = toSectionId;

    // Add to end of array
    sourceOrder.splice(indexSource, 1);
    sourceOrder.push(newFromMetric);

    // Save the new order and persist it
    setCurrentOrder(sourceOrder);
    localStorageStore.current.setItem(STORAGE_ITEM_KEY, JSON.stringify(sourceOrder));
  };

  const orderedMetricConfigs = useMemo(() => {
    if (currentOrder) {
      return orderBy(currentOrder, (config) => {
        const orderKey =
          currentOrder !== undefined && includesKey(currentOrder, config.metricKey)
            ? currentOrder.findIndex((chart) => chart.metricKey === config.metricKey)
            : Number.MAX_SAFE_INTEGER;
        return orderKey;
      });
    } else {
      return metricConfigs;
    }
  }, [currentOrder, metricConfigs]);

  return { orderedMetricConfigs, onReorderChart, onInsertChart };
};

export const useSectionExpanded = (sectionKeys: string[], storageNamespace: string, storageKey: string) => {
  const localStorageStore = useRef(LocalStorageUtils.getStoreForComponent(storageNamespace, storageKey));

  const [expandedSections, setExpandedSections] = useState<string[]>(() => {
    try {
      const sectionsExpanded = localStorageStore.current.getItem(SECTION_STORAGE_ITEM_KEY);
      if (sectionsExpanded) {
        return JSON.parse(sectionsExpanded);
      } else {
        localStorageStore.current.setItem(SECTION_STORAGE_ITEM_KEY, JSON.stringify(sectionKeys));
        return sectionKeys;
      }
    } catch {
      return {};
    }
  });

  useEffect(() => {
    localStorageStore.current.setItem(SECTION_STORAGE_ITEM_KEY, JSON.stringify(expandedSections));
  }, [expandedSections]);

  const onToggleSection = (key: string | string[]) => {
    const newExpandedSections = sectionKeys.filter((sectionKey: string) => {
      return (typeof key === 'string' && sectionKey === key) || (Array.isArray(key) && key.includes(sectionKey));
    });
    setExpandedSections(newExpandedSections);
  };

  return { expandedSections, onToggleSection };
};
