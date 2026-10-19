import { useState } from 'react';
import { Resizable } from 'react-resizable';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Button,
  ChevronRightIcon,
  ParagraphSkeleton,
  SidebarCollapseIcon,
  SidebarExpandIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { RunDetailsSidebar } from './RunDetailsSidebar';
import { useEvalRunSidebarData } from './useEvalRunSidebarData';
import { EvalRunComparisonsSection } from './EvalRunComparisonsSection';

const MIN_SIDEBAR_WIDTH = 320;
const MAX_SIDEBAR_WIDTH = 640;

export interface EvalRunTracesSidebarProps {
  experimentId: string;
  runUuid: string;
  runName?: string;
}

export const EvalRunTracesSidebar = ({ experimentId, runUuid, runName }: EvalRunTracesSidebarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [collapsed, setCollapsed] = useState(false);
  const [width, setWidth] = useState(MIN_SIDEBAR_WIDTH);
  const { sidebarProps, isLoading, error, isTruncated, hasPartialUsage } = useEvalRunSidebarData({
    experimentId,
    runUuid,
  });
  const toggleLabel = collapsed
    ? intl.formatMessage({
        defaultMessage: 'Show run details',
        description: 'Expand the evaluation run traces sidebar',
      })
    : intl.formatMessage({
        defaultMessage: 'Hide run details',
        description: 'Collapse the evaluation run traces sidebar',
      });

  return (
    <Resizable
      width={collapsed ? 0 : width}
      height={0}
      axis="x"
      resizeHandles={['w']}
      minConstraints={[MIN_SIDEBAR_WIDTH, 0]}
      maxConstraints={[MAX_SIDEBAR_WIDTH, 0]}
      draggableOpts={{ disabled: collapsed, cancel: 'button' }}
      onResize={(_event, { size }) => setWidth(size.width)}
      handle={
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label={intl.formatMessage({
            defaultMessage: 'Resize run details',
            description: 'Resize handle for the evaluation run sidebar',
          })}
          aria-valuemin={MIN_SIDEBAR_WIDTH}
          aria-valuemax={MAX_SIDEBAR_WIDTH}
          aria-valuenow={width}
          tabIndex={collapsed ? -1 : 0}
          onKeyDown={(event) => {
            if (collapsed || event.target !== event.currentTarget) return;
            if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
              event.preventDefault();
              setWidth((current) =>
                Math.min(
                  MAX_SIDEBAR_WIDTH,
                  Math.max(MIN_SIDEBAR_WIDTH, current + (event.key === 'ArrowLeft' ? 16 : -16)),
                ),
              );
            }
          }}
          css={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: collapsed ? 0 : -theme.spacing.sm,
            width: collapsed ? 0 : theme.spacing.md,
            zIndex: 1,
            cursor: collapsed ? 'default' : 'ew-resize',
            '& button': { opacity: 1 },
          }}
        >
          {!collapsed && (
            <Button
              componentId="mlflow.eval_run_details.sidebar.toggle"
              size="small"
              icon={<ChevronRightIcon />}
              aria-label={toggleLabel}
              title={toggleLabel}
              aria-expanded={!collapsed}
              aria-controls="eval-run-sidebar-content"
              onClick={() => setCollapsed((value) => !value)}
              css={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                // Native buttons enforce their visual styles with !important.
                '&&&': {
                  borderRadius: '50% !important',
                  height: `${theme.spacing.lg}px !important`,
                  border: `1px solid ${theme.colors.border} !important`,
                  backgroundColor: `${theme.colors.backgroundPrimary} !important`,
                  color: `${theme.colors.textSecondary} !important`,
                  '&:hover, &:active': {
                    backgroundColor: `${theme.colors.backgroundSecondary} !important`,
                  },
                  '& .anticon, & .db-icon': { color: 'inherit !important' },
                },
                width: theme.spacing.lg,
                minWidth: theme.spacing.lg,
                padding: 0,
              }}
            />
          )}
        </div>
      }
    >
      <section
        aria-label={intl.formatMessage({
          defaultMessage: 'Run sidebar',
          description: 'Evaluation run traces sidebar region',
        })}
        style={{ width: collapsed ? 0 : width }}
        css={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          maxWidth: '100%',
          borderLeft: `1px solid ${theme.colors.border}`,
          paddingLeft: collapsed ? 0 : theme.spacing.md,
          marginLeft: collapsed ? 0 : theme.spacing.md,
        }}
      >
        <div id="eval-run-sidebar-content" hidden={collapsed}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            <EvalRunComparisonsSection experimentId={experimentId} runUuid={runUuid} runName={runName ?? ''} />
            {isLoading ? (
              <ParagraphSkeleton
                label={intl.formatMessage({
                  defaultMessage: 'Loading run details',
                  description: 'Loading label for evaluation run sidebar',
                })}
              />
            ) : (
              <div>
                <RunDetailsSidebar {...sidebarProps} />
              </div>
            )}
            {error ? (
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Some run details could not be loaded."
                  description="Evaluation run sidebar data load failure"
                />
              </Typography.Text>
            ) : isTruncated || hasPartialUsage ? (
              <Typography.Text color="secondary">
                <FormattedMessage
                  defaultMessage="Stats include only loaded traces with recorded usage."
                  description="Evaluation run sidebar totals may be partial because traces or usage are missing"
                />
              </Typography.Text>
            ) : null}
          </div>
        </div>
        <div
          css={{
            marginTop: collapsed ? 0 : 'auto',
            paddingTop: collapsed ? 0 : theme.spacing.lg,
            ...(collapsed && {
              position: 'absolute',
              bottom: theme.spacing.sm,
              left: theme.spacing.xs,
              width: theme.general.heightSm,
              zIndex: 2,
            }),
          }}
        >
          <Button
            componentId="mlflow.eval_run_details.sidebar.footer_toggle"
            type="tertiary"
            icon={
              collapsed ? (
                <SidebarExpandIcon css={{ transform: 'rotate(180deg)' }} />
              ) : (
                <SidebarCollapseIcon css={{ transform: 'rotate(180deg)' }} />
              )
            }
            aria-label={
              collapsed
                ? intl.formatMessage({
                    defaultMessage: 'Expand run details',
                    description: 'Footer control to expand the evaluation run sidebar',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Collapse run details',
                    description: 'Footer control to collapse the evaluation run sidebar',
                  })
            }
            title={toggleLabel}
            aria-expanded={!collapsed}
            aria-controls="eval-run-sidebar-content"
            onClick={() => setCollapsed((value) => !value)}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              width: '100%',
              minHeight: theme.general.heightSm,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              boxSizing: 'border-box',
              borderRadius: theme.borders.borderRadiusSm,
              cursor: 'pointer',
              color: theme.colors.actionDefaultIconDefault,
              ':hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
              },
              '&&&': {
                color: `${theme.colors.textSecondary} !important`,
                justifyContent: `${collapsed ? 'center' : 'flex-start'} !important`,
                '& .anticon, & .db-icon': { color: 'inherit !important' },
                '&:hover, &:active': {
                  backgroundColor: `${theme.colors.backgroundSecondary} !important`,
                },
              },
            }}
          >
            {!collapsed && (
              <FormattedMessage
                defaultMessage="Collapse"
                description="Footer label for collapsing the evaluation run sidebar"
              />
            )}
          </Button>
        </div>
      </section>
    </Resizable>
  );
};
