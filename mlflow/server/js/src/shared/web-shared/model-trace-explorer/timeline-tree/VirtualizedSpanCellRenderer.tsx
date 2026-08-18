import {
  Button,
  Typography,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  GavelIcon,
  LinkIcon,
  Tooltip,
} from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useGatewayTraceLink } from '../hooks/useGatewayTraceLink';
import { Link } from '../RoutingUtils';
import type { HierarchyBar } from './TimelineTree.types';
import { TimelineTreeHierarchyBars } from './TimelineTreeHierarchyBars';
import { TimelineTreeSpanTooltip } from './TimelineTreeSpanTooltip';
import { TimelineTreeZIndex } from './TimelineTree.utils';

export interface VirtualizedSpanRow {
  id: string | number;
  node: ModelTraceSpanNode;
  isExpanded: boolean;
  hasChildren: boolean;
  isSelected: boolean;
  isInActiveChain: boolean;
  linesToRender: HierarchyBar[];
  onToggleExpand: (id: string | number) => void;
}

interface VirtualizedSpanCellRendererProps {
  data: VirtualizedSpanRow;
}

export const VirtualizedSpanCellRenderer = ({ data }: VirtualizedSpanCellRendererProps) => {
  const { theme } = useDesignSystemTheme();
  const { setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const { node, isExpanded, hasChildren, isSelected, isInActiveChain, linesToRender } = data;
  const hasException = getSpanExceptionCount(node) > 0;
  const gatewayTraceHref = useGatewayTraceLink(node.linkedGatewayTraceId);

  return (
    <TimelineTreeSpanTooltip span={node}>
      <div
        data-testid={`timeline-tree-node-${node.key}`}
        css={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          overflow: 'hidden',
          flex: 1,
          height: '100%',
          padding: `0px ${theme.spacing.sm}px`,
          boxSizing: 'border-box',
        }}
      >
        {hasChildren ? (
          <Button
            size="small"
            data-testid={`toggle-span-expanded-${node.key}`}
            css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
            icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={(event: React.MouseEvent) => {
              event.stopPropagation();
              data.onToggleExpand(data.id);
            }}
            componentId="shared.model-trace-explorer.virtualized-toggle-span"
          />
        ) : (
          <div css={{ width: 24, marginRight: theme.spacing.xs, flexShrink: 0 }} />
        )}
        <TimelineTreeHierarchyBars
          isActiveSpan={isSelected}
          isInActiveChain={isInActiveChain}
          linesToRender={linesToRender}
          hasChildren={hasChildren}
          isExpanded={isExpanded}
        />
        <span
          css={{
            flexShrink: 0,
            marginRight: theme.spacing.xs,
            borderRadius: theme.borders.borderRadiusSm,
            border: `1px solid ${isInActiveChain ? theme.colors.blue500 : theme.colors.backgroundSecondary}`,
            zIndex: TimelineTreeZIndex.NORMAL,
          }}
        >
          {node.icon}
        </span>
        <Typography.Text
          color={hasException ? 'error' : 'primary'}
          css={{
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            flex: 1,
          }}
        >
          {node.title}
        </Typography.Text>
        {gatewayTraceHref && (
          <Tooltip content="View linked gateway trace" componentId="shared.model-trace-explorer.gateway-trace-link">
            <Link
              componentId="mlflow.model_trace_explorer.timeline.gateway_trace_link"
              to={gatewayTraceHref}
              target="_blank"
              rel="noreferrer"
              data-testid={`gateway-trace-link-${node.key}`}
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
              css={{
                flexShrink: 0,
                display: 'flex',
                alignItems: 'center',
                marginLeft: theme.spacing.xs,
                color: theme.colors.actionPrimaryBackgroundDefault,
              }}
            >
              <LinkIcon css={{ fontSize: 14 }} />
            </Link>
          </Tooltip>
        )}
        {node.assessments.length > 0 && (
          <Tag
            color="indigo"
            data-testid={`assessment-tag-${node.key}`}
            componentId="shared.model-trace-explorer.assessment-count"
            css={{ margin: 0, borderRadius: theme.borders.borderRadiusSm }}
            onClick={() => setAssessmentsPaneExpanded?.(true)}
          >
            <GavelIcon />
            <Typography.Text css={{ marginLeft: theme.spacing.xs }}>{node.assessments.length}</Typography.Text>
          </Tag>
        )}
      </div>
    </TimelineTreeSpanTooltip>
  );
};
