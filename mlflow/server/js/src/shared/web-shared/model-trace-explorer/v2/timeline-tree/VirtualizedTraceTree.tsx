import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ColDef, GridApi, ICellRendererParams, RowHeightParams } from '@ag-grid-community/core';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { MLFlowAgGridLoader } from '../../../../../common/components/ag-grid/AgGridLoader';
import type { ModelTraceSpanNode, SpanFilterState } from '../ModelTrace.types';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import type { HierarchyBar } from './TimelineTree.types';
import { TimelineTreeHeader } from './TimelineTreeHeader';
import { getTimelineTreeMetricValue } from './TimelineTreeMetrics';
import { TimelineTreeNode } from './TimelineTreeNode';
import { SPAN_ROW_HEIGHT } from './TimelineTree.utils';

const SPAN_ROW_HEIGHT_WITH_METADATA = 48;

interface VirtualizedSpanRow {
  id: string | number;
  node: ModelTraceSpanNode;
  linesToRender: HierarchyBar[];
  expandedKeys: Set<string | number>;
  onSelect: (node: ModelTraceSpanNode) => void;
  selectedKey: string | number;
  setExpandedKeys: (keys: Set<string | number>) => void;
}

const collectAllKeys = (nodes: ModelTraceSpanNode[]): Set<string | number> => {
  const keys = new Set<string | number>();
  const walk = (node: ModelTraceSpanNode) => {
    keys.add(node.key);
    node.children?.forEach(walk);
  };
  nodes.forEach(walk);
  return keys;
};

const findAncestorPath = (nodes: ModelTraceSpanNode[], targetKey: string | number): Set<string | number> => {
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
};

const flattenTreeToRows = (
  rootNodes: ModelTraceSpanNode[],
  expandedKeys: Set<string | number>,
  onSelect: (node: ModelTraceSpanNode) => void,
  setExpandedKeys: (keys: Set<string | number>) => void,
  selectedKey?: string | number,
): VirtualizedSpanRow[] => {
  const activePath = selectedKey === undefined ? new Set<string | number>() : findAncestorPath(rootNodes, selectedKey);
  const rows: VirtualizedSpanRow[] = [];

  const walk = (node: ModelTraceSpanNode, parentLines: HierarchyBar[]) => {
    const children = node.children ?? [];
    rows.push({
      id: node.key,
      node,
      linesToRender: parentLines,
      expandedKeys,
      onSelect,
      selectedKey: selectedKey ?? '',
      setExpandedKeys,
    });

    if (children.length > 0 && expandedKeys.has(node.key)) {
      const activeChildIndex = children.findIndex((child) => activePath.has(child.key));
      children.forEach((child, index) => {
        walk(child, [
          ...parentLines,
          {
            shouldRender: index < children.length - 1,
            isActive: index < activeChildIndex,
          },
        ]);
      });
    }
  };

  rootNodes.forEach((node) => walk(node, []));
  return rows;
};

const SpanCellRenderer = ({ data }: ICellRendererParams): React.ReactElement | null => {
  if (!data) return null;

  const { node, linesToRender, expandedKeys, onSelect, selectedKey, setExpandedKeys } = data as VirtualizedSpanRow;
  return (
    <TimelineTreeNode
      node={node}
      selectedKey={selectedKey}
      expandedKeys={expandedKeys}
      setExpandedKeys={setExpandedKeys}
      traceStartTime={0}
      traceEndTime={0}
      onSelect={onSelect}
      linesToRender={linesToRender}
      renderChildren={false}
    />
  );
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
}): React.ReactElement => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { timelineTreeMetrics } = useModelTraceExplorerPreferences();
  const gridApiRef = useRef<GridApi | null>(null);
  const [expandedKeys, setExpandedKeys] = useState<Set<string | number>>(() => collectAllKeys(rootNodes));

  useEffect(() => {
    setExpandedKeys(collectAllKeys(rootNodes));
  }, [rootNodes]);

  const rowData = useMemo(
    () => flattenTreeToRows(rootNodes, expandedKeys, onSelect, setExpandedKeys, selectedNode?.key),
    [rootNodes, expandedKeys, onSelect, selectedNode?.key],
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

  const getRowHeight = useCallback(
    ({ data }: RowHeightParams): number => {
      const node = (data as VirtualizedSpanRow | undefined)?.node;
      if (!node || timelineTreeMetrics.length <= 1) return SPAN_ROW_HEIGHT;

      const hasVisibleMetric = timelineTreeMetrics.some(
        (metric) => getTimelineTreeMetricValue(metric, node, intl) !== undefined,
      );
      return hasVisibleMetric ? SPAN_ROW_HEIGHT_WITH_METADATA : SPAN_ROW_HEIGHT;
    },
    [intl, timelineTreeMetrics],
  );

  useEffect(() => {
    gridApiRef.current?.resetRowHeights();
    gridApiRef.current?.refreshCells({ force: true });
  }, [getRowHeight, rowData]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <TimelineTreeHeader
        showTimelineInfo={false}
        setShowTimelineInfo={() => {}}
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
          getRowId={(params) => String(params.data.id)}
          headerHeight={0}
          rowHeight={SPAN_ROW_HEIGHT}
          getRowHeight={getRowHeight}
          suppressHorizontalScroll
          suppressCellFocus
          onGridReady={(event) => {
            gridApiRef.current = event.api;
          }}
        />
      </div>
    </div>
  );
};
