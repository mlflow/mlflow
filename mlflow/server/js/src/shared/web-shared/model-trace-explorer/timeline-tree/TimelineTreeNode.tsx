import {
  Button,
  Typography,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  GavelIcon,
} from '@databricks/design-system';

import type { HierarchyBar } from './TimelineTree.types';
import { getActiveChildIndex, TimelineTreeZIndex } from './TimelineTree.utils';
import { TimelineTreeHierarchyBars } from './TimelineTreeHierarchyBars';
import { TimelineTreeSpanTooltip } from './TimelineTreeSpanTooltip';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';

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
}) => {
  const expanded = expandedKeys.has(node.key);
  const { theme } = useDesignSystemTheme();
  const hasChildren = (node.children ?? []).length > 0;
  const { setAssessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const isActive = selectedKey === node.key;
  const activeChildIndex = getActiveChildIndex(node, String(selectedKey));
  // true if a span has active children OR is the active span
  const isInActiveChain = activeChildIndex > -1;

  const hasException = getSpanExceptionCount(node) > 0;

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
            cursor: 'pointer',
            boxSizing: 'border-box',
            backgroundColor,
            ':hover': {
              backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            ':active': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
          }}
          onClick={() => {
            onSelect?.(node);
          }}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              // add padding to root nodes, because they have no connecting lines
              padding: `0px ${theme.spacing.sm}px`,
              justifyContent: 'space-between',
              overflow: 'hidden',
              flex: 1,
            }}
          >
            <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', overflow: 'hidden', flex: 1 }}>
              {hasChildren ? (
                <Button
                  size="small"
                  data-testid={`toggle-span-expanded-${node.key}`}
                  css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
                  icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
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
                <div css={{ width: 24, marginRight: theme.spacing.xs }} />
              )}
              <TimelineTreeHierarchyBars
                isActiveSpan={isActive}
                isInActiveChain={isInActiveChain}
                linesToRender={linesToRender}
                hasChildren={hasChildren}
                isExpanded={expanded}
              />
              <span
                css={{
                  flexShrink: 0,
                  marginRight: theme.spacing.xs,
                  borderRadius: theme.borders.borderRadiusSm,
                  border: `1px solid ${
                    activeChildIndex > -1 ? theme.colors.blue500 : theme.colors.backgroundSecondary
                  }`,
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
              {node.assessments.length > 0 && (
                <Tag
                  color="indigo"
                  data-testid={`assessment-tag-${node.key}`}
                  componentId="shared.model-trace-explorer.assessment-count"
                  css={{
                    margin: 0,
                    borderRadius: theme.borders.borderRadiusSm,
                  }}
                  onClick={() => setAssessmentsPaneExpanded?.(true)}
                >
                  <GavelIcon />
                  <Typography.Text css={{ marginLeft: theme.spacing.xs }}>{node.assessments.length}</Typography.Text>
                </Tag>
              )}
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
