import { createContext, useContext } from 'react';

// Generic, catalog-agnostic node selection. The host provides this around a
// TreeView surface so node clicks can drive host state (e.g. opening a span's
// inputs/outputs in a ContentViewer). The TreeView only reports an opaque node
// id; it has no knowledge of spans.
export type NodeSelectionContextValue = {
  enabled: boolean;
  selectedId?: string;
  onSelect?: (id: string) => void;
};

const NodeSelectionContext = createContext<NodeSelectionContextValue>({ enabled: false });

export const NodeSelectionProvider = NodeSelectionContext.Provider;

export const useNodeSelection = (): NodeSelectionContextValue => useContext(NodeSelectionContext);
