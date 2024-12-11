import { useMemo } from 'react';
import {
  ATTRIBUTE_COLUMN_SORT_KEY,
  ATTRIBUTE_COLUMN_SORT_LABEL,
  COLUMN_SORT_BY_ASC,
  COLUMN_SORT_BY_DESC,
  COLUMN_TYPES,
  SORT_DELIMITER_SYMBOL,
} from '../../../constants';
import { makeCanonicalSortKey } from '../utils/experimentPage.common-utils';

export type ExperimentRunSortOption = {
  label: string;
  order: string;
  value: string;
};

type SORT_KEY_TYPE = keyof (typeof ATTRIBUTE_COLUMN_SORT_KEY & typeof ATTRIBUTE_COLUMN_SORT_LABEL);

/**
 * This hook creates a set of run+sort options basing on currently selected
 * columns and the list of all metrics and keys.
 */
export const useRunSortOptions = (
  filteredMetricKeys: string[],
  filteredParamKeys: string[],
): ExperimentRunSortOption[] =>
  useMemo(() => {
    let sortOptions = [];
    const ColumnSortByOrder = [COLUMN_SORT_BY_ASC, COLUMN_SORT_BY_DESC];
    const attributesSortBy = Object.keys(ATTRIBUTE_COLUMN_SORT_LABEL).reduce<any[]>((options, sortLabelKey) => {
      const sortLabel = ATTRIBUTE_COLUMN_SORT_LABEL[sortLabelKey as SORT_KEY_TYPE];

      ColumnSortByOrder.forEach((order) => {
        options.push({
          label: sortLabel,
          value: ATTRIBUTE_COLUMN_SORT_KEY[sortLabelKey as SORT_KEY_TYPE] + SORT_DELIMITER_SYMBOL + order,
          order,
        });
      });

      return options;
    }, []);
    const metricsSortBy = filteredMetricKeys.reduce<any[]>((options, sortLabelKey) => {
      ColumnSortByOrder.forEach((order) => {
        options.push({
          label: sortLabelKey,
          value: `${makeCanonicalSortKey(COLUMN_TYPES.METRICS, sortLabelKey)}${SORT_DELIMITER_SYMBOL}${order}`,
          order,
        });
      });

      return options;
    }, []);
    const paramsSortBy = filteredParamKeys.reduce<any[]>((options, sortLabelKey) => {
      ColumnSortByOrder.forEach((order) => {
        options.push({
          label: sortLabelKey,
          value: `${makeCanonicalSortKey(COLUMN_TYPES.PARAMS, sortLabelKey)}${SORT_DELIMITER_SYMBOL}${order}`,
          order,
        });
      });

      return options;
    }, []);
    sortOptions = [...attributesSortBy, ...metricsSortBy, ...paramsSortBy];

    return sortOptions;
  }, [filteredMetricKeys, filteredParamKeys]);
