import {
  Button,
  CheckIcon,
  ChevronDownIcon,
  DropdownMenu,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import React, { useMemo, useState } from 'react';
import { getStableColorForRun as getStableColorForNode } from '../../../utils/RunNameUtils';
import { FormattedMessage } from 'react-intl';

type NodeSelectionState = 'none' | 'full' | 'partial';

/**
 * A node level metric charts node selector
 */
export const RunViewNodeLevelMetricChartsNodeSelector = ({
  nodesWithGpusConfig,
  selectedNodes,
  selectedGpus,
  onToggleNode,
  onToggleGpu,
  onClear,
}: {
  nodesWithGpusConfig: { nodeId: string; gpuCount: number }[];
  selectedNodes: Set<string>;
  selectedGpus: Map<string, Set<number>>;
  onToggleNode: (nodeId: string) => void;
  onToggleGpu: (nodeId: string, gpuIndex: number, totalGpuCount: number) => void;
  onClear?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);

  const selectionCounts = useMemo(() => {
    const gpuCount = Array.from(selectedGpus.values()).reduce((sum, set) => sum + set.size, 0);
    return { nodes: selectedNodes.size, gpus: gpuCount };
  }, [selectedNodes, selectedGpus]);

  const hasSelection = selectionCounts.nodes > 0 || selectionCounts.gpus > 0;

  const getNodeState = (nodeId: string): NodeSelectionState => {
    if (selectedNodes.has(nodeId)) return 'full';
    if (selectedGpus.has(nodeId)) return 'partial';
    return 'none';
  };

  return (
    <DropdownMenu.Root open={open} onOpenChange={setOpen}>
      <DropdownMenu.Trigger asChild>
        <Button componentId="mlflow.node-level-metric-charts.filter.trigger" endIcon={<ChevronDownIcon />}>
          <Typography.Text>
            {hasSelection ? (
              <>
                <FormattedMessage
                  defaultMessage="{nodeCount, plural, =0 {} one {# node} other {# nodes}}"
                  description="Count of selected nodes displayed in the node level metric charts node selector"
                  values={{
                    nodeCount: selectionCounts.nodes,
                  }}
                />
                {selectionCounts.nodes > 0 && selectionCounts.gpus > 0 && ', '}
                {selectionCounts.gpus > 0 && (
                  <FormattedMessage
                    defaultMessage="{gpuCount, plural, =0 {} one {# GPU} other {# GPUs}} selected"
                    description="Count of selected GPUs displayed in the node level metric charts node selector"
                    values={{ gpuCount: selectionCounts.gpus }}
                  />
                )}
              </>
            ) : (
              <FormattedMessage defaultMessage="Filter by node" description="Filter button label" />
            )}
          </Typography.Text>
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content>
        {nodesWithGpusConfig.map(({ nodeId, gpuCount }) => {
          const state = getNodeState(nodeId);
          const nodeLabel = (
            <FormattedMessage
              defaultMessage="Node {nodeId}"
              description="Label for a specific compute node in the node level metric charts node selector"
              values={{ nodeId }}
            />
          );
          return (
            <React.Fragment key={nodeId}>
              {gpuCount > 0 ? (
                <DropdownMenu.Sub>
                  <DropdownMenu.SubTrigger css={{ display: 'flex', gap: theme.spacing.sm }}>
                    <div
                      css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleNode(nodeId);
                      }}
                    >
                      <SelectionIndicator state={state} />
                      <NodeColorDot nodeId={nodeId} />
                      <Typography.Text>{nodeLabel}</Typography.Text>
                    </div>
                  </DropdownMenu.SubTrigger>
                  <DropdownMenu.SubContent>
                    {Array.from({ length: gpuCount }, (_, i) => (
                      <DropdownMenu.CheckboxItem
                        key={i}
                        componentId="mlflow.node-level-metric-charts.filter.by_gpu"
                        onClick={(e) => {
                          e.preventDefault();
                          onToggleGpu(nodeId, i, gpuCount);
                        }}
                        checked={state === 'full' || selectedGpus.get(nodeId)?.has(i)}
                      >
                        <DropdownMenu.ItemIndicator />
                        <Typography.Text>GPU {i}</Typography.Text>
                      </DropdownMenu.CheckboxItem>
                    ))}
                  </DropdownMenu.SubContent>
                </DropdownMenu.Sub>
              ) : (
                <DropdownMenu.CheckboxItem
                  componentId="mlflow.node-level-metric-charts.filter.by_node"
                  onClick={(e) => {
                    e.preventDefault();
                    onToggleNode(nodeId);
                  }}
                  checked={state === 'full'}
                >
                  <DropdownMenu.ItemIndicator />
                  <NodeColorDot nodeId={nodeId} />
                  <Typography.Text>{nodeLabel}</Typography.Text>
                </DropdownMenu.CheckboxItem>
              )}
            </React.Fragment>
          );
        })}
        {onClear && (
          <>
            <DropdownMenu.Separator />
            <DropdownMenu.Item
              componentId="mlflow.node-level-metric-charts.filter.clear"
              onClick={() => {
                onClear();
                setOpen(false);
              }}
            >
              <FormattedMessage defaultMessage="Clear filter" description="Clear filter button" />
            </DropdownMenu.Item>
          </>
        )}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

const NodeColorDot = ({ nodeId }: { nodeId: string }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        backgroundColor: getStableColorForNode(nodeId),
        borderRadius: '100%',
        width: 12,
        height: 12,
        border: `1px solid ${theme.colors.grey400}`,
      }}
    />
  );
};

const SelectionIndicator = ({ state }: { state: NodeSelectionState }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ width: 16, height: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {state === 'full' && <CheckIcon />}
      {state === 'partial' && (
        <div css={{ width: 10, height: 2, backgroundColor: theme.colors.actionPrimaryBackgroundDefault }} />
      )}
    </div>
  );
};
