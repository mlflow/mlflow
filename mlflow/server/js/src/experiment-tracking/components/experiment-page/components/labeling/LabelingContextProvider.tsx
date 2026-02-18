/**
 * Context provider for labeling session state management
 *
 * Simplified for OSS - excludes chat rounds and agent interaction.
 * Manages labeling progress and active task state.
 */

import React from 'react';

import type { LabelingItem, LabelingSession } from '../../../types/labeling';
import { useUpdateLabelingItem } from '../../../sdk/labeling';

export interface TaskRef {
  traceId: string;
  schemaName: string;
}

interface TaskState {
  completed: boolean;
  divRef?: React.RefObject<HTMLDivElement>;
}

interface LabelingContextType {
  // The active task that the user is currently working on
  activeTask: TaskRef | null;
  setActiveTask: (task: TaskRef | null) => void;

  // Set the active task to the next uncompleted task
  goToNextUncompletedTask: () => void;

  // Progress of the current item
  totalTaskCount: number;
  completedTaskCount: number;

  // Update the progress of the current item
  updateProgress: (task: TaskRef, completed: boolean) => void;

  // Used for the task to report its divRef to the context for scrolling
  setTaskDivRef: (task: TaskRef, divRef: React.RefObject<HTMLDivElement> | null) => void;

  // Mutation to update the item state based on labeling progress
  updateItemMutation: ReturnType<typeof useUpdateLabelingItem>;
}

const Context = React.createContext<LabelingContextType | null>(null);

export const useLabelingContext = () => {
  const context = React.useContext(Context);

  if (!context) {
    throw new Error('useLabelingContext must be used within <LabelingContextProvider>');
  }

  return context;
};

export const LabelingContextProvider = ({
  session,
  item,
  children,
}: {
  session: LabelingSession;
  item: LabelingItem;
  children: React.ReactNode;
}) => {
  // For OSS, we have a single trace per item (no chat rounds)
  const allTasks: TaskRef[] = React.useMemo(() => {
    if (!item.trace_id) return [];
    return session.labelingSchemas.map((schema) => ({
      traceId: item.trace_id!,
      schemaName: schema.name,
    }));
  }, [item.trace_id, session.labelingSchemas]);

  const [activeTask, setActiveTask] = React.useState<TaskRef | null>(null);

  // traceId -> schemaName -> TaskState
  const [taskState, setTaskState] = React.useState<Record<string, Record<string, TaskState>>>({});

  // Check if all task states are collected
  const allTaskStatesCollected = React.useMemo(() => {
    return allTasks.every((task) => taskState[task.traceId]?.[task.schemaName]?.completed != null);
  }, [allTasks, taskState]);

  const updateProgress = React.useCallback((task: TaskRef, completed: boolean) => {
    setTaskState((prevProgress) => {
      const updatedProgress = { ...prevProgress };
      if (!updatedProgress[task.traceId]) {
        updatedProgress[task.traceId] = {};
      }
      updatedProgress[task.traceId][task.schemaName] = {
        ...(updatedProgress[task.traceId][task.schemaName] ?? {}),
        completed,
      };
      return updatedProgress;
    });
  }, []);

  const setTaskDivRef = React.useCallback(
    (task: TaskRef, divRef: React.RefObject<HTMLDivElement> | null) => {
      if (!divRef) return;
      setTaskState((prevProgress) => {
        const updatedProgress = { ...prevProgress };
        if (!updatedProgress[task.traceId]) {
          updatedProgress[task.traceId] = {};
        }
        updatedProgress[task.traceId][task.schemaName] = {
          ...(updatedProgress[task.traceId][task.schemaName] ?? {}),
          divRef,
        };
        return updatedProgress;
      });
    },
    [],
  );

  const scrollToTask = React.useCallback(
    (task: TaskRef) => {
      const taskDivRef = taskState[task.traceId]?.[task.schemaName]?.divRef?.current;
      if (taskDivRef) {
        setTimeout(() => taskDivRef.scrollIntoView({ behavior: 'smooth' }), 100);
      }
    },
    [taskState],
  );

  const activeTaskIndex = activeTask
    ? allTasks.findIndex(
        (task) => task.traceId === activeTask.traceId && task.schemaName === activeTask.schemaName,
      )
    : -1;

  const goToNextUncompletedTask = React.useCallback(() => {
    const nextUncompletedTask = allTasks.find(
      (task, index) =>
        index > activeTaskIndex && !taskState[task.traceId]?.[task.schemaName]?.completed,
    );
    const nextTask = allTasks.find((_, index) => index > activeTaskIndex);
    const targetTask = nextUncompletedTask ?? nextTask;
    if (!targetTask) return;

    if (!activeTask || targetTask.traceId !== activeTask.traceId) {
      scrollToTask(targetTask);
    }
    setActiveTask(targetTask);
  }, [activeTask, activeTaskIndex, allTasks, scrollToTask, taskState]);

  // Auto go to next uncompleted task when the active task is not set
  const [hasAutoScrolled, setHasAutoScrolled] = React.useState(false);
  React.useEffect(() => {
    if (allTaskStatesCollected && !hasAutoScrolled && !activeTask) {
      goToNextUncompletedTask();
      setHasAutoScrolled(true);
    }
  }, [allTaskStatesCollected, hasAutoScrolled, activeTask, goToNextUncompletedTask]);

  const totalTaskCount = allTasks.length;
  const completedTaskCount = allTasks.filter(
    (task) => taskState[task.traceId]?.[task.schemaName]?.completed,
  ).length;

  // Update item state based on labeling progress
  const updateItemMutation = useUpdateLabelingItem();

  React.useEffect(() => {
    if (updateItemMutation.isPending) return;
    if (!allTaskStatesCollected) return;

    const targetState =
      totalTaskCount > 0 && completedTaskCount === totalTaskCount
        ? 'COMPLETED'
        : totalTaskCount > 0 && completedTaskCount > 0
          ? 'IN_PROGRESS'
          : 'PENDING';

    if (item.state !== targetState) {
      updateItemMutation.mutate({
        labeling_item_id: item.labeling_item_id,
        state: targetState,
      });
    }
  }, [
    allTaskStatesCollected,
    updateItemMutation,
    completedTaskCount,
    item.state,
    item.labeling_item_id,
    totalTaskCount,
  ]);

  return (
    <Context.Provider
      value={{
        activeTask,
        setActiveTask,
        goToNextUncompletedTask,
        totalTaskCount,
        completedTaskCount,
        updateProgress,
        setTaskDivRef,
        updateItemMutation,
      }}
    >
      {children}
    </Context.Provider>
  );
};
