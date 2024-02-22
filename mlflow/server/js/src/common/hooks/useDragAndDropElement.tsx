import { useEffect, useRef } from 'react';
import { useDrag, useDrop } from 'react-dnd';

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
  useEffect(() => {
    dropListener.current = onDrop;
  }, [onDrop]);
  const [{ isOver, draggedItem }, dropTargetRef] = useDrop<
    DraggedItem,
    never,
    { isOver: boolean; draggedItem: DraggedItem }
  >(
    () => ({
      canDrop: () => !disabled,
      accept: `dnd-${dragGroupKey}`,
      drop: ({ key: sourceKey }: { key: string }, monitor) => {
        if (sourceKey === dragKey || monitor.didDrop()) {
          return;
        }
        dropListener.current(sourceKey, dragKey);
      },
      collect: (monitor) => ({ isOver: monitor.isOver({ shallow: true }), draggedItem: monitor.getItem() }),
    }),
    [],
  );

  const [{ isDragging }, dragHandleRef, dragPreviewRef] = useDrag(
    () => ({
      canDrag: () => !disabled,
      type: `dnd-${dragGroupKey}`,
      item: { key: dragKey, groupKey: dragGroupKey },
      collect: (monitor) => ({
        isDragging: monitor.isDragging(),
      }),
    }),
    [],
  );

  const isDraggingOtherGroup = Boolean(draggedItem && draggedItem.groupKey !== dragGroupKey);

  return { dropTargetRef, dragHandleRef, dragPreviewRef, isDragging, isOver, isDraggingOtherGroup };
};
