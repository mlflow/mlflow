// eslint-disable-next-line import/no-extraneous-dependencies
import { type DragDropManager, createDragDropManager } from 'dnd-core';
import { useLayoutEffect, useRef, useState } from 'react';
import { useDrag, useDrop, DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

export interface useDragAndDropElementProps {
  /**
   * Unique drag element identifier, used in drop events
   */
  dragKey: string;
  /**
   * Group key - items are droppable only within a single group
   */
  dragGroupKey: string;
  /**
   * Callback function, conveys both dragged and target element key
   */
  onDrop: (draggedKey: string, targetDropKey: string) => void;

  /**
   * If true, drag and drop will be disabled
   */
  disabled?: boolean;
}

interface DraggedItem {
  key: string;
  groupKey: string;
}

/**
 * A hook enabling drag-and-drop capabilities for any component.
 * Used component will serve as both DnD source and target.
 */
export const useDragAndDropElement = ({
  dragGroupKey,
  dragKey,
  onDrop,
  disabled = false,
}: useDragAndDropElementProps) => {
  const dropListener = useRef(onDrop);

  dropListener.current = onDrop;

  const [{ isOver, draggedItem }, dropTargetRef] = useDrop<
    DraggedItem,
    never,
    { isOver: boolean; draggedItem: DraggedItem }
  >(
    {
      canDrop: () => !disabled,
      accept: `dnd-${dragGroupKey}`,
      drop: ({ key: sourceKey }: { key: string }, monitor) => {
        if (sourceKey === dragKey || monitor.didDrop()) {
          return;
        }
        dropListener.current(sourceKey, dragKey);
      },
      collect: (monitor) => ({ isOver: monitor.isOver({ shallow: true }), draggedItem: monitor.getItem() }),
    },
    [disabled, dragGroupKey, dragKey],
  );

  const [{ isDragging }, dragHandleRef, dragPreviewRef] = useDrag(
    {
      canDrag: () => !disabled,
      type: `dnd-${dragGroupKey}`,
      item: { key: dragKey, groupKey: dragGroupKey },
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
      }),
    },
    [disabled, dragGroupKey, dragKey],
  );

  const isDraggingOtherGroup = Boolean(draggedItem && draggedItem.groupKey !== dragGroupKey);

  return { dropTargetRef, dragHandleRef, dragPreviewRef, isDragging, isOver, isDraggingOtherGroup };
};

/**
 * Create a scoped DndProvider that will limit its functionality to a single root element.
 * It should prevent HTML5Backend collisions, see:
 * https://github.com/react-dnd/react-dnd/blob/7c88c37489a53b5ac98699c46a506a8e085f1c03/packages/backend-html5/src/HTML5BackendImpl.ts#L107-L109
 */
export const DragAndDropProvider: React.FC<React.PropsWithChildren<unknown>> = ({ children }) => {
  const rootElementRef = useRef<HTMLDivElement>(null);
  const [manager, setManager] = useState<DragDropManager | null>(null);

  useLayoutEffect(() => {
    const rootElement = rootElementRef.current;
    const dragDropManager = createDragDropManager(HTML5Backend, undefined, { rootElement });
    setManager(dragDropManager);
    return () => {
      dragDropManager.getBackend().teardown();
    };
  }, []);

  return (
    <div css={{ display: 'contents' }} ref={rootElementRef}>
      {manager && <DndProvider manager={manager}>{children}</DndProvider>}
    </div>
  );
};
