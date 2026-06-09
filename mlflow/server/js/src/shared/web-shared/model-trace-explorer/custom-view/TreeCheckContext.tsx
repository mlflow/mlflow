import { createContext, useContext } from 'react';

// Generic, catalog-agnostic multi-selection for TreeView rows. The host provides
// this around a TreeView surface so per-node checkboxes can drive host state
// (e.g. collecting the spans to scope "Add feedback" to). The TreeView only
// reports opaque node ids; it has no knowledge of spans.
export type TreeCheckContextValue = {
  enabled: boolean;
  checkedIds: Set<string>;
  onToggle?: (id: string, checked: boolean) => void;
};

const TreeCheckContext = createContext<TreeCheckContextValue>({ enabled: false, checkedIds: new Set() });

export const TreeCheckProvider = TreeCheckContext.Provider;

export const useTreeCheck = (): TreeCheckContextValue => useContext(TreeCheckContext);
