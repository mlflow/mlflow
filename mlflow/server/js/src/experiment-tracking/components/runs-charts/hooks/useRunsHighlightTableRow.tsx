import { type RefObject, useCallback, useEffect } from 'react';
import { ChartsTraceHighlightSource, useRunsChartTraceHighlight } from './useRunsChartTraceHighlight';
import type { CellMouseOverEvent } from '@ag-grid-community/core';

const DEFAULT_HIGH_LIGHT_CLASS_NAME = 'is-highlighted';

/**
 * Helper hook adding support for useRunsChartTraceSetHighlight() logic to a ag-grid table rows
 */
export const useRunsHighlightTableRow = (
  /**
   * Reference to the container element of the table.
   */
  containerElementRef: RefObject<HTMLDivElement>,
  /**
   * Class name to be added to the highlighted row.
   */
  highlightedClassName = DEFAULT_HIGH_LIGHT_CLASS_NAME,
  /**
   * Additional selector prefix to be used to find the row element.
   */
  findInFlexColumns = false,
  /**
   * Optional function to extract the row UUID from the table data, used in row hover callback.
   */
  getRowUuid?: (data: any) => string | undefined,
) => {
  const { onHighlightChange, highlightDataTrace } = useRunsChartTraceHighlight();
  /**
   * Listener function that highlights a row in the table by adding a class to it.
   */
  const highlightFn = useCallback(
    (rowUuid: string | null, source?: ChartsTraceHighlightSource) => {
      // First, quickly remove the highlight class from the previous highlighted row
      const existingHighlightedRowElement = containerElementRef.current?.querySelector(`.${highlightedClassName}`);

      const additionalSelectorPrefix = findInFlexColumns ? '.ag-center-cols-viewport' : '';

      // Find the new row element and add the highlight class to it
      const rowElement = containerElementRef.current?.querySelector(
        `${additionalSelectorPrefix} .ag-row[row-id="${rowUuid}"]`,
      );
      if (existingHighlightedRowElement && existingHighlightedRowElement !== rowElement) {
        existingHighlightedRowElement.classList.remove(highlightedClassName);
      }

      // Do not highlight the row if the source of highlight event is the table itself
      if (source === ChartsTraceHighlightSource.TABLE) {
        return;
      }

      rowElement && rowElement.classList.add(highlightedClassName);
    },
    [containerElementRef, highlightedClassName, findInFlexColumns],
  );

  // Subscribe to the highlight change event
  useEffect(() => onHighlightChange(highlightFn), [highlightFn, onHighlightChange]);

  // Create event handlers for table cell mouse over and out events
  const cellMouseOverHandler = useCallback(
    ({ data }: CellMouseOverEvent) => {
      const isGroupRow = typeof data === 'object' && 'groupParentInfo' in data;
      // Extract the trace UUID from the data
      // Use runUuid for non-group rows and rowUuid for group rows
      const dataTraceUuid = getRowUuid ? getRowUuid({ data }) : isGroupRow ? data.rowUuid : data.runUuid;

      highlightDataTrace(dataTraceUuid, {
        source: ChartsTraceHighlightSource.TABLE,
      });
    },
    [highlightDataTrace, getRowUuid],
  );

  const cellMouseOutHandler = useCallback(() => highlightDataTrace(null), [highlightDataTrace]);

  return { cellMouseOverHandler, cellMouseOutHandler };
};
