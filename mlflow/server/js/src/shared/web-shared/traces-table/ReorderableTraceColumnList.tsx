import type { DragEndEvent } from '@dnd-kit/core';
import { closestCenter, DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToParentElement, restrictToVerticalAxis } from '@dnd-kit/modifiers';
import {
  sortableKeyboardCoordinates,
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Button, DragIcon, DropdownMenu, useDesignSystemTheme } from '@databricks/design-system';

import type { GenericColumnOption } from './TraceColumnSelector';

export interface ReorderableTraceColumnOption extends GenericColumnOption {
  reorderLabel: string;
  /** Locks visibility while leaving the column available for reordering. */
  disabled?: boolean;
}

export interface ReorderableTraceColumnListProps {
  columns: ReorderableTraceColumnOption[];
  visibleColumns: string[];
  onToggleColumn: (column: string) => void;
  onReorderColumn: (activeColumn: string, targetColumn: string) => void;
}

interface ReorderableTraceColumnItemProps {
  column: Omit<ReorderableTraceColumnOption, 'componentId'>;
  componentId: string;
  checked: boolean;
  index: number;
  columnOrder: string[];
  onToggleColumn: (column: string) => void;
  onReorderColumn: (activeColumn: string, targetColumn: string) => void;
}

const ReorderableTraceColumnItem = ({
  column,
  componentId,
  checked,
  index,
  columnOrder,
  onToggleColumn,
  onReorderColumn,
}: ReorderableTraceColumnItemProps) => {
  const { theme } = useDesignSystemTheme();
  const { attributes, listeners, setActivatorNodeRef, setNodeRef, transform, transition, isDragging } = useSortable({
    id: column.id,
  });

  const handleKeyboardReorder = (event: React.KeyboardEvent) => {
    if (!event.ctrlKey || (event.key !== 'ArrowUp' && event.key !== 'ArrowDown')) {
      return;
    }
    const targetIndex = event.key === 'ArrowUp' ? index - 1 : index + 1;
    const targetColumn = columnOrder[targetIndex];
    if (targetColumn === undefined) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    onReorderColumn(column.id, targetColumn);
  };

  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      css={{
        display: 'flex',
        alignItems: 'center',
        opacity: isDragging ? 0.5 : 1,
      }}
    >
      <DropdownMenu.CheckboxItem
        componentId={componentId}
        checked={checked}
        disabled={column.disabled}
        aria-keyshortcuts="Control+ArrowUp Control+ArrowDown"
        css={{ flex: 1 }}
        onKeyDown={handleKeyboardReorder}
        onSelect={(event) => {
          event.preventDefault();
          onToggleColumn(column.id);
        }}
      >
        <DropdownMenu.ItemIndicator />
        {column.label}
      </DropdownMenu.CheckboxItem>
      <Button
        componentId="web-shared.traces-table.column-reorder.handle"
        size="small"
        type="tertiary"
        icon={<DragIcon css={{ color: theme.colors.textSecondary }} />}
        aria-label={column.reorderLabel}
        ref={setActivatorNodeRef}
        css={{
          cursor: 'grab',
          color: theme.colors.textSecondary,
          flexShrink: 0,
          touchAction: 'none',
          '&:active': { cursor: 'grabbing' },
        }}
        onClick={(event) => event.stopPropagation()}
        {...attributes}
        {...listeners}
      />
    </div>
  );
};

/** A checkable column list with pointer-drag and Ctrl+Arrow keyboard reordering. */
export const ReorderableTraceColumnList = ({
  columns,
  visibleColumns,
  onToggleColumn,
  onReorderColumn,
}: ReorderableTraceColumnListProps): React.ReactElement => {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const visible = new Set(visibleColumns);
  const columnOrder = columns.map(({ id }) => id);

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (over !== null && active.id !== over.id && typeof active.id === 'string' && typeof over.id === 'string') {
      onReorderColumn(active.id, over.id);
    }
  };

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      modifiers={[restrictToVerticalAxis, restrictToParentElement]}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={columnOrder} strategy={verticalListSortingStrategy}>
        {columns.map(({ componentId, ...column }, index) => (
          <ReorderableTraceColumnItem
            key={column.id}
            column={column}
            componentId={componentId}
            checked={column.disabled ? true : visible.has(column.id)}
            index={index}
            columnOrder={columnOrder}
            onToggleColumn={onToggleColumn}
            onReorderColumn={onReorderColumn}
          />
        ))}
      </SortableContext>
    </DndContext>
  );
};
