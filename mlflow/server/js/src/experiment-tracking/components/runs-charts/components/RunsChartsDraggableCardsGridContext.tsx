/* eslint-disable react-hooks/rules-of-hooks */
import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';
import type { RunsChartsCardConfig } from '../runs-charts.types';
import { DragAndDropProvider } from '../../../../common/hooks/useDragAndDropElement';
import { useUpdateRunsChartsUIConfiguration } from '../hooks/useRunsChartsUIConfiguration';
import { indexOf, sortBy } from 'lodash';

const RunsChartsDraggableGridStateContext = createContext<{
  draggedCardUuid: string | null;
  targetSection: string | null;
  isDragging: () => boolean;
}>({
  draggedCardUuid: '',
  targetSection: '',
  isDragging: () => false,
});

const RunsChartsDraggableGridActionsContext = createContext<{
  setDraggedCardUuid: (cardUuid: string | null) => void;
  setTargetSection: (targetSectionUuid: string | null) => void;
  setTargetPosition: (pos: number) => void;
  onDropChartCard: () => void;
  onSwapCards: (sourceUuid: string, targetUuid: string) => void;
}>({
  setDraggedCardUuid: () => {},
  setTargetSection: () => {},
  setTargetPosition: () => {},
  onDropChartCard: () => {},
  onSwapCards: () => {},
});

export const useRunsChartsDraggableGridStateContext = () => useContext(RunsChartsDraggableGridStateContext);
export const useRunsChartsDraggableGridActionsContext = () => useContext(RunsChartsDraggableGridActionsContext);

export const RunsChartsDraggableCardsGridContextProvider = ({
  children,
  visibleChartCards = [],
}: {
  children?: React.ReactNode;
  visibleChartCards?: RunsChartsCardConfig[];
}) => {
  // Stateful values: dragged card ID and target section ID
  const [draggedCardUuid, setDraggedCardUuid] = useState<string | null>(null);
  const [targetSectionUuid, setTargetSectionUuid] = useState<string | null>(null);

  // Use refs for direct access to the values
  const immediateDraggedCardUuid = useRef<string | null>(null);
  const immediateTargetSectionId = useRef<string | null>(null);
  immediateDraggedCardUuid.current = draggedCardUuid;
  immediateTargetSectionId.current = targetSectionUuid;

  // Mutable field: target position (index) in the target section
  const targetLocalPosition = useRef<number | null>(null);

  const setTargetPosition = useCallback((pos: number) => {
    targetLocalPosition.current = pos;
  }, []);

  const isDragging = useCallback(() => immediateDraggedCardUuid.current !== null, []);

  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();

  // Callback for simple card swapping (Move up/move down functionality)
  const onSwapCards = useCallback(
    (sourceChartUuid: string, targetChartUuid: string) => {
      updateChartsUIState((current) => {
        const newChartsOrder = current.compareRunCharts?.slice();

        if (!newChartsOrder) {
          return current;
        }

        const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
        const indexTarget = newChartsOrder.findIndex((c) => c.uuid === targetChartUuid);

        // If one of the charts is not found, do nothing
        if (indexSource < 0 || indexTarget < 0) {
          return current;
        }

        const sourceChart = newChartsOrder[indexSource];

        // Remove the chart and insert it at the target position
        newChartsOrder.splice(indexSource, 1);
        newChartsOrder.splice(indexTarget, 0, sourceChart);

        return { ...current, compareRunCharts: newChartsOrder };
      });
    },
    [updateChartsUIState],
  );

  // Callback invoked when a card is dropped
  const onDropChartCard = useCallback(() => {
    const draggedChartUuid = immediateDraggedCardUuid.current;
    const targetSectionId = immediateTargetSectionId.current;
    const targetPosition = targetLocalPosition.current;

    if (draggedChartUuid === null || targetSectionId === null || targetPosition === null) {
      return;
    }

    setDraggedCardUuid(null);

    updateChartsUIState((current) => {
      // Find the source card and section in the current state
      const sourceCard = current.compareRunCharts?.find((chart) => chart.uuid === draggedChartUuid);
      if (!sourceCard) {
        return current;
      }

      const sourceSection = current.compareRunSections?.find((section) => section.uuid === sourceCard?.metricSectionId);

      // Find the target section in the current state
      const targetSection = current.compareRunSections?.find((section) => section.uuid === targetSectionId);

      // Get all the cards in the source section
      const sourceSectionCards = current.compareRunCharts?.filter(
        (chartCard) => chartCard.metricSectionId === sourceSection?.uuid,
      );

      // If we're moving the card within the same section
      if (sourceSection === targetSection) {
        // Copy the resulting chart list
        const resultChartsList = current.compareRunCharts?.slice();

        // Get all the currently visible cards (excluding hidden and deleted cards)
        const visibleSectionCards = sourceSectionCards?.filter((chartCard) => visibleChartCards.includes(chartCard));

        if (!resultChartsList || !visibleSectionCards) {
          return current;
        }

        // Find the original position
        const originalIndex = resultChartsList.findIndex((chartCard) => chartCard.uuid === draggedChartUuid);

        // Clamp the target position index to the visible section cards
        const clampedLocalPosition = Math.min(targetPosition, visibleSectionCards.length - 1);

        // Map from the index in the section to a global index
        const targetIndex = indexOf(resultChartsList, visibleSectionCards?.[clampedLocalPosition]);

        // If we found the original index and the target index
        if (resultChartsList && originalIndex !== -1 && targetIndex !== -1) {
          // Remove the card from the original position
          resultChartsList.splice(originalIndex, 1);
          // Insert the card at the target position
          resultChartsList.splice(targetIndex, 0, sourceCard);
        }

        return {
          ...current,
          compareRunCharts: resultChartsList,
        };
      } else {
        // If we're moving card to a new section
        const targetSectionCards = current.compareRunCharts?.filter(
          (chart) => chart.metricSectionId === targetSectionId && !chart.deleted,
        );

        // Calculate the target position in the target section
        if (targetSectionCards) {
          targetSectionCards.splice(targetPosition, 0, sourceCard);
        }

        return {
          ...current,
          // Use the target position in the sorting function to determine the new position
          compareRunCharts: sortBy(current.compareRunCharts, (a) => targetSectionCards?.indexOf(a) ?? -1).map(
            (chart) => {
              // Also, update the metricSectionId of the dragged card
              if (chart.uuid === sourceCard.uuid) {
                return {
                  ...chart,
                  metricSectionId: targetSectionId,
                };
              }
              return chart;
            },
          ),
          compareRunSections: current.compareRunSections?.map((section) => {
            if (section.uuid === sourceSection?.uuid || section.uuid === targetSection?.uuid) {
              return { ...section, isReordered: true };
            }
            return section;
          }),
        };
      }
    });
  }, [updateChartsUIState, visibleChartCards]);

  // For performance purposes, expose two different contexts:
  // one for state (changing but rarely consumed) and one for actions (static but consumed often)
  const stateContextValue = useMemo(
    () => ({
      draggedCardUuid,
      targetSection: targetSectionUuid,
      isDragging,
    }),
    [draggedCardUuid, targetSectionUuid, isDragging],
  );

  const actionsContextValue = useMemo(
    () => ({
      setDraggedCardUuid,
      setTargetSection: setTargetSectionUuid,
      setTargetPosition,
      onDropChartCard,
      onSwapCards,
    }),
    [onDropChartCard, onSwapCards, setTargetPosition],
  );

  return (
    <DragAndDropProvider>
      <RunsChartsDraggableGridStateContext.Provider value={stateContextValue}>
        <RunsChartsDraggableGridActionsContext.Provider value={actionsContextValue}>
          {children}
        </RunsChartsDraggableGridActionsContext.Provider>
      </RunsChartsDraggableGridStateContext.Provider>
    </DragAndDropProvider>
  );
};
