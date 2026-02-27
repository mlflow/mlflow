import { useLayoutEffect, useRef } from 'react';

import { useDesignSystemTheme, Typography, Button, ChevronDownIcon, ChevronRightIcon } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../../ModelTrace.types';
import { spanTimeFormatter, TimelineTreeZIndex } from '../TimelineTree.utils';
import { TimelineTreeSpanTooltip } from '../TimelineTreeSpanTooltip';

export const TimelineTreeGanttNode = ({
  node,
  selectedKey,
  leftOffset,
  width,
  onSelect,
  expandedKeys,
  setExpandedKeys,
}: {
  node: ModelTraceSpanNode;
  selectedKey: string | number;
  leftOffset: number;
  width: number;
  onSelect: ((node: ModelTraceSpanNode) => void) | undefined;
  expandedKeys: Set<string | number>;
  setExpandedKeys: (keys: Set<string | number>) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const contentInBarRef = useRef<HTMLSpanElement>(null);
  const contentBesideBarRef = useRef<HTMLSpanElement>(null);
  const isActive = selectedKey === node.key;
  const backgroundColor = isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent';
  const hasChildren = (node.children ?? []).length > 0;
  const expanded = expandedKeys.has(node.key);

  useLayoutEffect(() => {
    const contentInBar = contentInBarRef.current;
    const contentBesideBar = contentBesideBarRef.current;

    if (!contentInBar || !contentBesideBar) {
      return;
    }

    const showAllElements = () => {
      contentInBar.style.display = 'inline-flex';
      contentBesideBar.style.display = 'inline-flex';
    };

    const setDisplayStyles = (showInBar: boolean) => {
      contentInBar.style.display = showInBar ? 'inline-flex' : 'none';
      contentBesideBar.style.display = showInBar ? 'none' : 'inline-flex';
    };

    showAllElements();

    const contentWidth = Math.max(contentInBar.offsetWidth, contentBesideBar.offsetWidth);
    const fitsInBar = contentWidth < width - theme.spacing.sm * 2;

    setDisplayStyles(fitsInBar);
  }, [theme.spacing.sm, width, expanded, node.title, node.icon]);

  return (
    <TimelineTreeSpanTooltip span={node}>
      <div
        key={node.key}
        css={{
          display: 'flex',
          flexDirection: 'row',
          cursor: 'pointer',
          boxSizing: 'border-box',
          paddingLeft: theme.spacing.xs,
          paddingRight: theme.spacing.sm,
          paddingTop: theme.spacing.xs,
          paddingBottom: theme.spacing.xs,
          backgroundColor: backgroundColor,
          alignItems: 'center',
          ':hover': {
            backgroundColor: theme.colors.actionDefaultBackgroundHover,
          },
          ':active': {
            backgroundColor: theme.colors.actionDefaultBackgroundPress,
          },
          zIndex: TimelineTreeZIndex.NORMAL,
        }}
        onClick={() => onSelect?.(node)}
      >
        {hasChildren ? (
          <Button
            size="small"
            data-testid={`toggle-timeline-span-expanded-${node.key}`}
            css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
            icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={(event) => {
              event.stopPropagation();
              const newExpandedKeys = new Set(expandedKeys);
              if (expanded) {
                newExpandedKeys.delete(node.key);
              } else {
                newExpandedKeys.add(node.key);
              }
              setExpandedKeys(newExpandedKeys);
            }}
            componentId="shared.model-trace-explorer.toggle-timeline-span"
          />
        ) : (
          <div css={{ width: theme.spacing.lg, marginRight: theme.spacing.xs }} />
        )}
        <div css={{ width: leftOffset, flexShrink: 0 }} />
        <div
          css={{
            position: 'relative',
            width,
            height: theme.typography.lineHeightLg,
            backgroundColor: theme.isDarkMode ? theme.colors.blue600 : theme.colors.blue300,
            border: `1px solid ${theme.isDarkMode ? theme.colors.blue400 : theme.colors.blue500}`,
            borderRadius: theme.borders.borderRadiusSm,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Typography.Text color="primary" css={{ display: 'flex', alignItems: 'center' }}>
            <span
              ref={contentInBarRef}
              css={{
                display: 'none',
                alignItems: 'center',
              }}
            >
              {node.icon && (
                <span
                  css={{
                    flexShrink: 0,
                    '& > div': {
                      height: theme.general.buttonInnerHeight,
                      borderTopRightRadius: 0,
                      borderBottomRightRadius: 0,
                    },
                  }}
                >
                  {node.icon}
                </span>
              )}
              <span
                css={{
                  marginLeft: theme.spacing.xs,
                  color: theme.colors.textPrimary,
                }}
              >
                {node.title}
              </span>
            </span>
          </Typography.Text>
        </div>
        <div
          css={{
            flex: 1,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            marginLeft: theme.spacing.sm,
          }}
        >
          <div
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              backgroundColor: theme.colors.backgroundSecondary,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusSm,
              paddingRight: theme.spacing.xs,
            }}
          >
            <Typography.Text color="primary" css={{ display: 'flex', alignItems: 'center' }}>
              <span
                ref={contentBesideBarRef}
                css={{
                  display: 'none',
                  alignItems: 'center',
                }}
              >
                {node.icon && (
                  <span
                    css={{
                      flexShrink: 0,
                      marginLeft: 0,
                      borderRight: `1px solid ${theme.colors.border}`,
                      '& > div': {
                        height: theme.general.buttonInnerHeight,
                        borderTopRightRadius: 0,
                        borderBottomRightRadius: 0,
                      },
                    }}
                  >
                    {node.icon}
                  </span>
                )}
                <span
                  css={{
                    color: theme.colors.textPrimary,
                    marginLeft: theme.spacing.xs,
                  }}
                >
                  {node.title}
                </span>
              </span>
            </Typography.Text>
            <Typography.Text color="secondary">
              <span css={{ marginLeft: theme.spacing.xs }}>{spanTimeFormatter(node.end - node.start)}</span>
            </Typography.Text>
          </div>
        </div>
      </div>
    </TimelineTreeSpanTooltip>
  );
};
