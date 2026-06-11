import { createContext, useContext } from 'react';

// Depth of a TreeNode within the tree, used for indentation. TreeView seeds 0
// at the root; each TreeNode bumps it by one for the children it renders.
export const TreeDepthContext = createContext<number>(0);

export const useTreeDepth = (): number => useContext(TreeDepthContext);

// An author-supplied directive describing one entry in a node's side panel. The
// host turns these into real components (KeyValueViewer / Markdown /
// FeedbackButtons) on selection, pulling span data from its nodeMap — so the
// author/LLM never has to emit the heavy span inputs/outputs themselves.
export type PanelItem = {
  type: 'input' | 'output' | 'attributes' | 'markdown' | 'feedback';
  // markdown body (type === 'markdown').
  text?: string;
  // section heading (markdown) or KeyValueViewer label override.
  title?: string;
  // feedback prompt text (type === 'feedback').
  label?: string;
  // feedback assessment name (type === 'feedback').
  name?: string;
};

/**
 * Selection + span-deeplink coordination for a single TreeView. The TreeView
 * owns the selection state and a `spanId -> node` registry; TreeNodes consume
 * it to (a) highlight the selected row, (b) trigger the host to build that
 * node's side panel from its `panelItems`, and (c) register the span id they
 * represent so markdown `#span:<id>` deeplinks can select the owning node.
 */
export type TreeSelectionContextValue = {
  enabled: boolean;
  selectedNodeId?: string;
  // The component id the host injects the built side-panel subtree at; TreeView
  // renders it via buildChild.
  selectedPanelId?: string;
  // Select a node: updates state and asks the host (via an A2UI action) to build
  // the node's side panel from `panelItems` + the span's data.
  select: (nodeId: string, panelItems: PanelItem[], spanId?: string) => void;
  // TreeNodes register the span they represent (+ their panel directives) so
  // deeplinks can resolve and rebuild the right panel.
  registerSpan: (spanId: string, entry: { nodeId: string; panelItems: PanelItem[] }) => void;
  unregisterSpan: (spanId: string) => void;
  // Select the node that represents `spanId` (used by markdown deeplinks).
  selectSpan: (spanId: string) => void;
};

const noop = () => {};

const TreeSelectionContext = createContext<TreeSelectionContextValue>({
  enabled: false,
  select: noop,
  registerSpan: noop,
  unregisterSpan: noop,
  selectSpan: noop,
});

export const TreeSelectionProvider = TreeSelectionContext.Provider;

export const useTreeSelection = (): TreeSelectionContextValue => useContext(TreeSelectionContext);
