import { RowSelectionState } from '@tanstack/react-table';
import { createContext, SetStateAction, useContext, useState } from 'react';

const GenAiTraceTableRowSelectionContext = createContext<{
  rowSelection: RowSelectionState;
  setRowSelection: (rowSelection: SetStateAction<RowSelectionState>) => void;
} | null>(null);

export const useGenAiTraceTableRowSelection = () => {
  const context = useContext(GenAiTraceTableRowSelectionContext);

  // In a regular use case, we use the local state to manage the row selection.
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  // However, if context is provided, we use the context to manage the row selection.
  if (context) {
    return context;
  }
  return { rowSelection, setRowSelection };
};

/**
 * Hook to check if we're inside a GenAiTraceTableRowSelectionProvider.
 * Useful for components that need to conditionally create their own provider.
 */
export const useIsInsideGenAiTraceTableRowSelectionProvider = () => {
  const context = useContext(GenAiTraceTableRowSelectionContext);
  return context !== null;
};

/**
 * Use this provider to manage selected rows across the table using a context.
 * If not used, the consumers of `useGenAiTraceTableRowSelection()` will use the local state to manage the row selection.
 */
export const GenAiTraceTableRowSelectionProvider = ({
  children,
  rowSelection,
  setRowSelection,
}: {
  children: React.ReactNode;
  rowSelection: RowSelectionState;
  setRowSelection: (rowSelection: SetStateAction<RowSelectionState>) => void;
}) => {
  return (
    <GenAiTraceTableRowSelectionContext.Provider value={{ rowSelection, setRowSelection }}>
      {children}
    </GenAiTraceTableRowSelectionContext.Provider>
  );
};
