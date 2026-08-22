import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import type { ColDef, GridApi, ICellRendererParams } from '@ag-grid-community/core';

import { MLFlowAgGridLoader } from '../../../../common/components/ag-grid/AgGridLoader';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { VirtualizedSpanCellRenderer } from './VirtualizedSpanCellRenderer';
import type { VirtualizedSpanRow } from './VirtualizedSpanCellRenderer';
import type { HierarchyBar } from './TimelineTree.types';
import { SPAN_ROW_HEIGHT } from './TimelineTree.utils';
import { TimelineTreeHeader } from './TimelineTreeHeader';

function collectAllKeys(nodes: ModelTraceSpanNode[]): Set<string | number> {
  const keys = new Set<string | number>();
  const walk = (node: ModelTraceSpanNode) => {
    keys.add(node.key);
    node.children?.forEach(walk);
  };
  nodes.forEach(walk);
  return keys;
}

function findAncestorPath(nodes: ModelTraceSpanNode[], targetKey: string | number): Set<string | number> {
  const path = new Set<string | number>();

  const search = (node: ModelTraceSpanNode): boolean => {
    if (node.key === targetKey) {
      path.add(node.key);
      return true;
    }
    for (const child of node.children ?? []) {
      if (search(child)) {
        path.add(node.key);
        return true;
      }
    }
    return false;
  };

  nodes.forEach(search);
  return path;
}

function flattenTreeToRows(
  rootNodes: ModelTraceSpanNode[],
  expandedKeys: Set<string | number>,
  selectedKey?: string | number,
): VirtualizedSpanRow[] {
  const activePath = selectedKey != null ? findAncestorPath(rootNodes, selectedKey) : new Set<string | number>();
  const rows: VirtualizedSpanRow[] = [];

  const walk = (node: ModelTraceSpanNode, depth: number, parentLines: HierarchyBar[]) => {
    const children = node.children ?? [];
    const hasChildren = children.length > 0;
    const isExpanded = expandedKeys.has(node.key);
    const isSelected = node.key === selectedKey;
    const isInActiveChain = activePath.has(node.key);

    rows.push({
      id: node.key,
      node,
      depth,
      isExpanded,
      hasChildren,
      isSelected,
      isInActiveChain,
      linesToRender: parentLines,
    });

    if (hasChildren && isExpanded) {
      const activeChildIdx = children.findIndex((c) => activePath.has(c.key));
      children.forEach((child, idx) => {
        const isLastChild = idx === children.length - 1;
        const childLines: HierarchyBar[] = [
          ...parentLines,
          { shouldRender: !isLastChild, isActive: idx < activeChildIdx },
        ];
        walk(child, depth + 1, childLines);
      });
    }
  };

  rootNodes.forEach((node) => walk(node, 0, []));
  return rows;
}

const SpanCellRenderer = ({
  data,
  context,
}: ICellRendererParams & { context: { onToggleExpand: (id: string | number) => void } }) => {
  if (!data) return null;
  return <VirtualizedSpanCellRenderer data={data as VirtualizedSpanRow} onToggleExpand={context.onToggleExpand} />;
};

export const VirtualizedTraceTree = ({
  rootNodes,
  selectedNode,
  onSelect,
  spanFilterState,
  setSpanFilterState,
}: {
  rootNodes: ModelTraceSpanNode[];
  selectedNode?: ModelTraceSpanNode;
  onSelect: (node: ModelTraceSpanNode) => void;
  spanFilterState: SpanFilterState;
  setSpanFilterState: (state: SpanFilterState) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const gridApiRef = useRef<GridApi | null>(null);
  const [showTimelineInfo, setShowTimelineInfo] = useState(false);

  const prevRootNodesRef = useRef(rootNodes);
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(() => collectAllKeys(rootNodes));

  if (prevRootNodesRef.current !== rootNodes) {
    prevRootNodesRef.current = rootNodes;
    setExpandedKeys(collectAllKeys(rootNodes));
  }

  const handleToggleExpand = useCallback((id: string | number) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const rowData = useMemo(
    () => flattenTreeToRows(rootNodes, expandedKeys, selectedNode?.key),
    [rootNodes, expandedKeys, selectedNode?.key],
  );

  const columnDefs = useMemo<ColDef[]>(
    () => [
      {
        field: 'id',
        flex: 1,
        cellRendererFramework: SpanCellRenderer,
      },
    ],
    [],
  );

  const gridContext = useMemo(() => ({ onToggleExpand: handleToggleExpand }), [handleToggleExpand]);

  const getRowStyle = useCallback(
    (params: any) => {
      if (params.data?.isSelected) {
        return { backgroundColor: theme.colors.actionDefaultBackgroundHover };
      }
      return undefined;
    },
    [theme.colors.actionDefaultBackgroundHover],
  );

  useEffect(() => {
    gridApiRef.current?.refreshCells({ force: true });
  }, [rowData]);

  const handleRowClicked = useCallback(
    (event: any) => {
      if (event.data) {
        onSelect((event.data as VirtualizedSpanRow).node);
      }
    },
    [onSelect],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <TimelineTreeHeader
        showTimelineInfo={showTimelineInfo}
        setShowTimelineInfo={setShowTimelineInfo}
        spanFilterState={spanFilterState}
        setSpanFilterState={setSpanFilterState}
        hideTimelineToggle
      />
      <div
        css={{
          flex: 1,
          minHeight: 0,
          '.ag-row': { cursor: 'pointer', border: 'none !important' },
          '.ag-row:hover': { backgroundColor: `${theme.colors.actionDefaultBackgroundHover} !important` },
          '.ag-row:active': { backgroundColor: `${theme.colors.actionDefaultBackgroundPress} !important` },
          '.ag-cell': {
            display: 'flex',
            alignItems: 'center',
            padding: '0 !important',
            border: 'none !important',
          },
        }}
      >
        <MLFlowAgGridLoader
          rowData={rowData}
          columnDefs={columnDefs}
          context={gridContext}
          getRowId={(params) => String(params.data.id)}
          headerHeight={0}
          rowHeight={SPAN_ROW_HEIGHT}
          suppressHorizontalScroll
          suppressCellFocus
          onGridReady={(e) => {
            gridApiRef.current = e.api;
          }}
          onRowClicked={handleRowClicked}
          getRowStyle={getRowStyle}
        />
      </div>
    </div>
  );
};
