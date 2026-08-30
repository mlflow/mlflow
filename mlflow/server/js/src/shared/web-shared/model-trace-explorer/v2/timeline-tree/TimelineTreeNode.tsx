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
import { useIntl } from '@databricks/i18n';

import type { HierarchyBar } from './TimelineTree.types';
import { getActiveChildIndex, SPAN_ROW_HEIGHT, TimelineTreeZIndex } from './TimelineTree.utils';
import { TimelineTreeHierarchyBars } from './TimelineTreeHierarchyBars';
import { TimelineTreeSpanTooltip } from './TimelineTreeSpanTooltip';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useGatewayTraceLink } from '../../hooks/useGatewayTraceLink';
import { Link } from '../../RoutingUtils';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { getTimelineTreeMetricValue } from './TimelineTreeMetrics';

const ROW_HEIGHT = SPAN_ROW_HEIGHT;
const ROW_HEIGHT_WITH_METADATA = 48;
const MetadataItem = ({
  children,
  icon,
  title,
}: {
  children: React.ReactNode;
  icon?: React.ReactNode;
  title: string;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <span
      title={title}
      css={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        minWidth: 0,
        maxWidth: 160,
        overflow: 'hidden',
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {icon && (
        <span
          css={{
            display: 'inline-flex',
            alignItems: 'center',
            flexShrink: 0,
            color: theme.colors.textSecondary,
            '& svg': {
              fontSize: theme.typography.fontSizeSm - 1,
            },
          }}
        >
          {icon}
        </span>
      )}
      <Typography.Text size="sm" color="secondary" ellipsis>
        <span css={{ fontSize: theme.typography.fontSizeSm - 1 }}>{children}</span>
      </Typography.Text>
    </span>
  );
};

export const TimelineTreeNode = ({
  node,
  selectedKey,
  expandedKeys,
  setExpandedKeys,
  traceStartTime,
  traceEndTime,
  onSelect,
  linesToRender,
}: {
  node: ModelTraceSpanNode;
  selectedKey: string | number;
  expandedKeys: Set<string | number>;
  setExpandedKeys: (keys: Set<string | number>) => void;
  traceStartTime: number;
  traceEndTime: number;
  onSelect: ((node: ModelTraceSpanNode) => void) | undefined;
  // a boolean array that signifies whether or not a vertical
  // connecting line is supposed to in at the `i`th spacer. see
  // TimelineTreeHierarchyBars for more details.
  linesToRender: Array<HierarchyBar>;
}): React.ReactElement => {
  const expanded = expandedKeys.has(node.key);
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { timelineTreeMetrics } = useModelTraceExplorerPreferences();
  const hasChildren = (node.children ?? []).length > 0;
  const { setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const isActive = selectedKey === node.key;
  const activeChildIndex = getActiveChildIndex(node, String(selectedKey));
  // true if a span has active children OR is the active span
  const isInActiveChain = activeChildIndex > -1;

  const hasException = getSpanExceptionCount(node) > 0;
  const gatewayTraceHref = useGatewayTraceLink(node.linkedGatewayTraceId);
  const metricValues = timelineTreeMetrics.flatMap((metric) => {
    const value = getTimelineTreeMetricValue(metric, node, intl);
    return value ? [value] : [];
  });
  const inlineMetric = timelineTreeMetrics.length === 1 ? metricValues[0] : undefined;
  const hasMetricRow = timelineTreeMetrics.length > 1 && metricValues.length > 0;
  const rowHeight = hasMetricRow ? ROW_HEIGHT_WITH_METADATA : ROW_HEIGHT;
  const rowTopPadding = (rowHeight - 24) / 2;

  const backgroundColor = isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent';

  return (
    <>
      <TimelineTreeSpanTooltip span={node}>
        <div
          data-testid={`timeline-tree-node-${node.key}`}
          css={{
            display: 'flex',
            flexDirection: 'column',
            width: '100%',
            height: rowHeight,
            cursor: 'pointer',
            boxSizing: 'border-box',
            backgroundColor,
            ':hover': {
              backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            ':active': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
            position: 'relative',
          }}
          onClick={() => {
            onSelect?.(node);
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'stretch',
              // add padding to root nodes, because they have no connecting lines
              padding: `0px ${theme.spacing.sm}px`,
              justifyContent: 'space-between',
              overflow: 'hidden',
              flex: 1,
              height: rowHeight,
            }}
          >
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'stretch',
                overflow: 'hidden',
                flex: 1,
              }}
            >
              {hasChildren ? (
                <Button
                  size="small"
                  type="tertiary"
                  data-testid={`toggle-span-expanded-${node.key}`}
                  css={{ flexShrink: 0, marginRight: theme.spacing.xs, marginTop: rowTopPadding }}
                  icon={
                    expanded ? (
                      <ChevronDownIcon css={{ color: theme.colors.textSecondary, opacity: 0.7 }} />
                    ) : (
                      <ChevronRightIcon css={{ color: theme.colors.textSecondary, opacity: 0.7 }} />
                    )
                  }
                  onClick={(event) => {
                    // prevent the node from being selected when the expand button is clicked
                    event.stopPropagation();
                    const newExpandedKeys = new Set(expandedKeys);
                    if (expanded) {
                      newExpandedKeys.delete(node.key);
                    } else {
                      newExpandedKeys.add(node.key);
                    }
                    setExpandedKeys(newExpandedKeys);
                  }}
                  componentId="shared.model-trace-explorer.toggle-span"
                />
              ) : (
                <div css={{ width: 24, marginRight: theme.spacing.xs, marginTop: rowTopPadding, flexShrink: 0 }} />
              )}
              <TimelineTreeHierarchyBars
                isActiveSpan={isActive}
                isInActiveChain={isInActiveChain}
                linesToRender={linesToRender}
                hasChildren={hasChildren}
                isExpanded={expanded}
                rowHeight={rowHeight}
              />
              <div
                css={{
                  display: 'flex',
                  flex: 1,
                  minWidth: 0,
                  height: rowHeight,
                  boxSizing: 'border-box',
                  alignItems: 'center',
                  paddingTop: theme.spacing.xs,
                  paddingBottom: theme.spacing.xs,
                }}
              >
                <span
                  data-testid={`span-icon-${node.key}`}
                  css={{
                    flexShrink: 0,
                    marginRight: theme.spacing.sm,
                    borderRadius: theme.borders.borderRadiusSm,
                    border: `1px solid ${
                      activeChildIndex > -1 ? theme.colors.blue500 : theme.colors.backgroundSecondary
                    }`,
                    zIndex: TimelineTreeZIndex.NORMAL,
                    '& > div': {
                      width: theme.general.buttonInnerHeight,
                      height: theme.general.buttonInnerHeight,
                      '& svg': {
                        width: theme.typography.fontSizeBase,
                        height: theme.typography.fontSizeBase,
                      },
                    },
                  }}
                >
                  {node.icon}
                </span>
                <div
                  data-testid={`span-text-block-${node.key}`}
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    flex: 1,
                    minWidth: 0,
                  }}
                >
                  <div
                    data-testid={`span-title-row-${node.key}`}
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      minWidth: 0,
                      minHeight: 24,
                    }}
                  >
                    <div
                      css={{
                        display: 'flex',
                        alignItems: 'center',
                        flex: inlineMetric ? 1 : '0 1 auto',
                        minWidth: 0,
                        overflow: 'hidden',
                      }}
                    >
                      <Typography.Text
                        color={hasException ? 'error' : 'primary'}
                        css={{
                          overflow: 'hidden',
                          whiteSpace: 'nowrap',
                          textOverflow: 'ellipsis',
                          minWidth: 0,
                        }}
                      >
                        {node.title}
                      </Typography.Text>
                    </div>
                    {gatewayTraceHref && (
                      <Tooltip
                        content="View linked gateway trace"
                        componentId="shared.model-trace-explorer.gateway-trace-link"
                      >
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
                        css={{
                          margin: 0,
                          marginLeft: theme.spacing.xs,
                          borderRadius: theme.borders.borderRadiusSm,
                        }}
                        onClick={() => setAssessmentsPaneExpanded?.(true)}
                      >
                        <GavelIcon />
                        <Typography.Text css={{ marginLeft: theme.spacing.xs }}>
                          {node.assessments.length}
                        </Typography.Text>
                      </Tag>
                    )}
                    {inlineMetric && (
                      <div
                        data-testid={`span-inline-metric-${node.key}`}
                        css={{ flexShrink: 0, marginLeft: theme.spacing.xs }}
                      >
                        <MetadataItem title={inlineMetric.title} icon={inlineMetric.icon}>
                          {inlineMetric.value}
                        </MetadataItem>
                      </div>
                    )}
                  </div>
                  {hasMetricRow && (
                    <div
                      data-testid={`span-metric-row-${node.key}`}
                      css={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: theme.spacing.xs,
                        flexWrap: 'nowrap',
                        minWidth: 0,
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {metricValues.map((metric) => (
                        <MetadataItem key={metric.key} title={metric.title} icon={metric.icon}>
                          {metric.value}
                        </MetadataItem>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </TimelineTreeSpanTooltip>
      {expanded &&
        node.children?.map((child, idx) => (
          <TimelineTreeNode
            key={child.key}
            node={child}
            expandedKeys={expandedKeys}
            setExpandedKeys={setExpandedKeys}
            selectedKey={selectedKey}
            traceStartTime={traceStartTime}
            traceEndTime={traceEndTime}
            onSelect={onSelect}
            linesToRender={linesToRender.concat({
              // render the connecting line at this depth
              // if there are more children to render
              shouldRender: idx < (node.children?.length ?? 0) - 1,
              // make the vertical line blue if the active span
              // is below this child
              isActive: idx < activeChildIndex,
            })}
          />
        ))}
    </>
  );
};
