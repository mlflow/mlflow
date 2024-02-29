import {
  Button,
  DragIcon,
  DropdownMenu,
  OverflowIcon,
  Typography,
  useDesignSystemTheme,
  InfoTooltip,
  FullscreenIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { PropsWithChildren, ReactNode } from 'react';
import { useDragAndDropElement } from '../../../../../common/hooks/useDragAndDropElement';
import {
  shouldEnableDeepLearningUI,
  shouldEnableDeepLearningUIPhase2,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import { FormattedMessage } from 'react-intl';
import { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsCardConfig } from '../../runs-charts.types';

export enum RunsChartsChartsDragGroup {
  PARALLEL_CHARTS_AREA = 'PARALLEL_CHARTS_AREA',
  GENERAL_AREA = 'GENERAL_AREA',
}

export interface RunsChartCardReorderProps {
  onReorderWith: (draggedKey: string, targetDropKey: string) => void;
  canMoveUp: boolean;
  canMoveDown: boolean;
  onMoveUp: () => void;
  onMoveDown: () => void;
}

export type RunsChartCardSetFullscreenFn = (chart: {
  config: RunsChartsCardConfig;
  title: string;
  subtitle: ReactNode;
}) => void;

export interface RunsChartCardFullScreenProps {
  fullScreen?: boolean;
  setFullScreenChart?: RunsChartCardSetFullscreenFn;
}

export interface ChartCardWrapperProps extends RunsChartCardReorderProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  onEdit: () => void;
  onDelete: () => void;
  tooltip?: string;
  uuid?: string;
  dragGroupKey: RunsChartsChartsDragGroup;
  additionalMenuContent?: React.ReactNode;
  toggleFullScreenChart?: () => void;
}

export const ChartRunsCountIndicator = ({ runsOrGroups }: { runsOrGroups: RunsChartsRunData[] }) => {
  const containsGroups = runsOrGroups.some(({ groupParentInfo }) => groupParentInfo);
  const containsRuns = runsOrGroups.some(({ runInfo }) => runInfo);

  // After moving to the new run rows visibility model, we don't configure run count per chart
  if (shouldUseNewRunRowsVisibilityModel()) {
    return null;
  }

  return containsRuns && containsGroups ? (
    <FormattedMessage
      defaultMessage="Comparing first {count} groups and runs"
      values={{ count: runsOrGroups.length }}
      description="Experiment page > compare runs > chart header > compared groups and runs count label"
    />
  ) : containsGroups ? (
    <FormattedMessage
      defaultMessage="Comparing first {count} groups"
      values={{ count: runsOrGroups.length }}
      description="Experiment page > compare runs > chart header > compared groups count label"
    />
  ) : (
    <FormattedMessage
      defaultMessage="Comparing first {count} runs"
      values={{ count: runsOrGroups.length }}
      description="Experiment page > compare runs > chart header > compared runs count label"
    />
  );
};

/**
 * Wrapper components for all chart cards. Provides styles and adds
 * a dropdown menu with actions for configure and delete.
 */
export const RunsChartCardWrapper = ({
  title,
  subtitle,
  onDelete,
  onEdit,
  children,
  uuid,
  dragGroupKey,
  tooltip = '',
  onReorderWith = () => {},
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  additionalMenuContent,
  toggleFullScreenChart,
}: PropsWithChildren<ChartCardWrapperProps>) => {
  const { theme } = useDesignSystemTheme();
  const isDragAndDropEnabled = shouldEnableDeepLearningUI();

  const { dragHandleRef, dragPreviewRef, dropTargetRef, isDraggingOtherGroup, isOver } = useDragAndDropElement({
    dragGroupKey,
    dragKey: uuid || '',
    onDrop: onReorderWith,
    disabled: !isDragAndDropEnabled,
  });

  return (
    <div
      css={{
        height: 360,
        overflow: 'hidden',
        display: 'grid',
        gridTemplateRows: 'auto 1fr',
        backgroundColor: theme.colors.backgroundPrimary,
        padding: isDragAndDropEnabled ? 12 : theme.spacing.md,
        // have a slightly smaller padding when the enableDeepLearningUI
        // flag is on to accomodate the legend in the charts
        paddingBottom: isDragAndDropEnabled ? theme.spacing.sm : theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        transition: 'opacity 0.12s',
      }}
      style={{
        opacity: isDraggingOtherGroup ? 0.1 : isOver ? 0.5 : 1,
      }}
      data-testid="experiment-view-compare-runs-card"
      ref={(element) => {
        // Use this element for both drag preview and drop target
        dragPreviewRef?.(element);
        dropTargetRef?.(element);
      }}
    >
      <div
        css={{
          display: 'grid',
          // one extra column to accomodate the drag handle
          gridTemplateColumns: isDragAndDropEnabled ? 'auto 1fr auto auto' : '1fr auto auto',
        }}
      >
        {isDragAndDropEnabled && (
          <div
            ref={dragHandleRef}
            data-testid="experiment-view-compare-runs-card-drag-handle"
            css={{
              marginTop: theme.spacing.xs,
              marginRight: theme.spacing.sm,
              cursor: 'grab',
            }}
          >
            <DragIcon />
          </div>
        )}
        <div css={{ overflow: 'hidden', flex: 1 }}>
          <Typography.Title
            title={String(title)}
            level={4}
            css={{
              marginBottom: 0,
              overflow: 'hidden',
              whiteSpace: 'nowrap',
              textOverflow: 'ellipsis',
            }}
          >
            {title}
          </Typography.Title>
          {subtitle && <span css={styles.subtitle(theme)}>{subtitle}</span>}
          {tooltip && <InfoTooltip css={{ verticalAlign: 'middle' }} title={tooltip} />}
        </div>
        {shouldEnableDeepLearningUIPhase2() && (
          <Button
            componentId="fullscreen_button_chartcard"
            icon={<FullscreenIcon />}
            onClick={toggleFullScreenChart}
            disabled={!toggleFullScreenChart}
          />
        )}
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-compare_cards_chartcard.common.tsx_158"
              type="tertiary"
              icon={<OverflowIcon />}
              data-testid="experiment-view-compare-runs-card-menu"
            />
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="end" minWidth={100}>
            <DropdownMenu.Item onClick={onEdit} data-testid="experiment-view-compare-runs-card-edit">
              Configure
            </DropdownMenu.Item>
            <DropdownMenu.Item onClick={onDelete} data-testid="experiment-view-compare-runs-card-delete">
              Delete
            </DropdownMenu.Item>
            {isDragAndDropEnabled && (
              <>
                <DropdownMenu.Separator />
                <DropdownMenu.Item
                  disabled={!canMoveUp}
                  onClick={onMoveUp}
                  data-testid="experiment-view-compare-runs-move-up"
                >
                  <FormattedMessage
                    defaultMessage="Move up"
                    description="Experiment page > compare runs tab > chart header > move up option"
                  />
                </DropdownMenu.Item>
                <DropdownMenu.Item
                  disabled={!canMoveDown}
                  onClick={onMoveDown}
                  data-testid="experiment-view-compare-runs-move-down"
                >
                  <FormattedMessage
                    defaultMessage="Move down"
                    description="Experiment page > compare runs tab > chart header > move down option"
                  />
                </DropdownMenu.Item>
              </>
            )}
            {additionalMenuContent}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
      {children}
    </div>
  );
};

const styles = {
  chartEntry: (theme: Theme) => ({
    height: 360,
    overflow: 'hidden',
    display: 'grid',
    gridTemplateRows: 'auto 1fr',
    backgroundColor: theme.colors.backgroundPrimary,
    padding: theme.spacing.md,
    border: `1px solid ${theme.colors.border}`,
    borderRadius: theme.general.borderRadiusBase,
  }),
  chartComponentWrapper: () => ({
    overflow: 'hidden',
  }),
  subtitle: (theme: Theme) => ({
    color: theme.colors.textSecondary,
    fontSize: 11,
    marginRight: 4,
    verticalAlign: 'middle',
  }),
};
