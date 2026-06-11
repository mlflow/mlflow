import { useCallback, useMemo, useRef, useState } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, ChildListSchema, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { Button, CloseIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import {
  type PanelItem,
  type TreeSelectionContextValue,
  TreeDepthContext,
  TreeSelectionProvider,
} from './TreeSelectionContext';

// A2UI client action fired when a node is selected. The host's action handler
// matches on this name, builds the side-panel subtree from the node's
// panelItems + the span's data, and injects it via updateComponents.
export const TREE_NODE_SELECTED = 'TREE_NODE_SELECTED';

/**
 * Schema (API) for the TreeView container. It lays out a collapsible tree of
 * `TreeNode` children (referenced by id) on the left, and — when a node is
 * selected — renders the host-built side panel for that node on the right.
 * Selection is also driven by markdown `#span:<id>` deeplinks inside any node's
 * (or the panel's) markdown.
 */
export const TreeViewApi = {
  name: 'TreeView',
  schema: z
    .object({
      title: DynamicStringSchema.describe('Optional heading shown above the tree.').optional(),
      children: ChildListSchema.describe('The root TreeNode component ids, in display order.'),
      emptyMessage: DynamicStringSchema.describe('Text shown when there are no nodes.').optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const TreeView = createComponentImplementation(TreeViewApi, ({ props, context, buildChild }) => {
  const { theme } = useDesignSystemTheme();

  const childIds = useMemo(() => (Array.isArray(props.children) ? (props.children as string[]) : []), [props.children]);
  const title = props.title ? asString(props.title) : undefined;
  const emptyMessage = props.emptyMessage ? asString(props.emptyMessage) : 'No nodes to display.';

  const [selected, setSelected] = useState<{ nodeId?: string; panelId?: string }>({});
  // spanId -> node entry (+ panel directives), populated by TreeNodes so
  // #span:<id> deeplinks resolve and rebuild the right panel.
  const spanRegistry = useRef(new Map<string, { nodeId: string; panelItems: PanelItem[] }>());

  // Keep the latest component context in a ref so the (stable) callbacks below
  // can dispatch actions without changing identity on every render.
  const contextRef = useRef(context);
  contextRef.current = context;

  const select = useCallback((nodeId: string, panelItems: PanelItem[], spanId?: string) => {
    setSelected({ nodeId, panelId: `${nodeId}__panel` });
    void contextRef.current.dispatchAction({
      event: {
        name: TREE_NODE_SELECTED,
        context: { nodeId, ...(spanId ? { spanId } : {}), panelItems },
      },
    });
  }, []);
  const registerSpan = useCallback((spanId: string, entry: { nodeId: string; panelItems: PanelItem[] }) => {
    spanRegistry.current.set(spanId, entry);
  }, []);
  const unregisterSpan = useCallback((spanId: string) => {
    spanRegistry.current.delete(spanId);
  }, []);
  const selectSpan = useCallback(
    (spanId: string) => {
      const entry = spanRegistry.current.get(spanId);
      if (entry) {
        select(entry.nodeId, entry.panelItems, spanId);
      }
    },
    [select],
  );

  const selectionValue = useMemo<TreeSelectionContextValue>(
    () => ({
      enabled: true,
      selectedNodeId: selected.nodeId,
      selectedPanelId: selected.panelId,
      select,
      registerSpan,
      unregisterSpan,
      selectSpan,
    }),
    [selected.nodeId, selected.panelId, select, registerSpan, unregisterSpan, selectSpan],
  );

  return (
    <TreeSelectionProvider value={selectionValue}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        {title && (
          <Typography.Text bold size="lg">
            {title}
          </Typography.Text>
        )}
        <div css={{ display: 'flex', alignItems: 'stretch', gap: theme.spacing.md }}>
          <div css={{ flex: 1, minWidth: 0, maxHeight: 480, overflowY: 'auto' }}>
            {childIds.length === 0 ? (
              <Typography.Text color="secondary">{emptyMessage}</Typography.Text>
            ) : (
              <TreeDepthContext.Provider value={0}>
                {childIds.map((id) => (
                  <div key={id}>{buildChild(id)}</div>
                ))}
              </TreeDepthContext.Provider>
            )}
          </div>
          {selected.panelId && (
            <div
              css={{
                flex: 1,
                minWidth: 0,
                maxHeight: 480,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.sm,
                borderLeft: `1px solid ${theme.colors.border}`,
                paddingLeft: theme.spacing.md,
              }}
            >
              <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                <Button
                  componentId="shared.model-trace-explorer.custom-view.tree-view.close-panel"
                  size="small"
                  icon={<CloseIcon />}
                  aria-label="Close panel"
                  onClick={() => setSelected({})}
                />
              </div>
              {buildChild(selected.panelId)}
            </div>
          )}
        </div>
      </div>
    </TreeSelectionProvider>
  );
});
