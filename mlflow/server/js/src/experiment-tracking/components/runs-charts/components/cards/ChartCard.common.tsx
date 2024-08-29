import {
  Button,
  DragIcon,
  DropdownMenu,
  OverflowIcon,
  Typography,
  useDesignSystemTheme,
  LegacyInfoTooltip,
  FullscreenIcon,
  Switch,
  Spinner,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { PropsWithChildren, ReactNode, memo, useCallback, useRef } from 'react';
import { useDragAndDropElement } from '../../../../../common/hooks/useDragAndDropElement';
import {
  shouldEnableDraggableChartsGridLayout,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../common/utils/FeatureUtils';
import { FormattedMessage } from 'react-intl';
import { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsCardConfig } from '../../runs-charts.types';
import type { ExperimentChartImageDownloadFileFormat } from '../../hooks/useChartImageDownloadHandler';
import { noop } from 'lodash';

export const DRAGGABLE_CARD_HANDLE_CLASS = 'drag-handle';
export const DRAGGABLE_CARD_TRANSITION_NAME = '--drag-transform';
export const DRAGGABLE_CARD_TRANSITION_VAR = `var(${DRAGGABLE_CARD_TRANSITION_NAME})`;

export enum RunsChartsChartsDragGroup {
  PARALLEL_CHARTS_AREA = 'PARALLEL_CHARTS_AREA',
  GENERAL_AREA = 'GENERAL_AREA',
}

export interface RunsChartCardReorderProps {
  onReorderWith: (draggedKey: string, targetDropKey: string) => void;
  canMoveUp: boolean;
  canMoveDown: boolean;
  previousChartUuid?: string;
  nextChartUuid?: string;
}

export interface RunsChartCardSizeProps {
  height?: number;
  positionInSection?: number;
}

export interface RunsChartCardVisibilityProps {
  isInViewport?: boolean;
  isInViewportDeferred?: boolean;
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

export interface ChartCardToggleProps {
  toggleLabel: string;
  currentToggle: boolean;
  setToggle: () => void;
}

export interface ChartCardWrapperProps extends RunsChartCardReorderProps, RunsChartCardSizeProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  onEdit: () => void;
  onDelete: () => void;
  tooltip?: React.ReactNode;
  uuid?: string;
  dragGroupKey: RunsChartsChartsDragGroup;
  additionalMenuContent?: React.ReactNode;
  toggleFullScreenChart?: () => void;
  toggles?: ChartCardToggleProps[];
  isRefreshing?: boolean;
  onClickDownload?: (format: ExperimentChartImageDownloadFileFormat | 'csv' | 'csv-full') => void;
  supportedDownloadFormats?: (ExperimentChartImageDownloadFileFormat | 'csv' | 'csv-full')[];
  isHidden?: boolean;
}

export const ChartRunsCountIndicator = memo(({ runsOrGroups }: { runsOrGroups: RunsChartsRunData[] }) => {
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
});

/**
 * Wrapper components for all chart cards. Provides styles and adds
 * a dropdown menu with actions for configure and delete.
 */
export const RunsChartCardWrapperRaw = ({
  title,
  subtitle,
  onDelete,
  onEdit,
  children,
  uuid,
  dragGroupKey,
  tooltip = '',
  onReorderWith = noop,
  canMoveDown,
  canMoveUp,
  previousChartUuid,
  nextChartUuid,
  additionalMenuContent,
  toggleFullScreenChart,
  toggles,
  supportedDownloadFormats = [],
  onClickDownload,
  isHidden,
  height,
  isRefreshing = false,
}: PropsWithChildren<ChartCardWrapperProps>) => {
  const { theme } = useDesignSystemTheme();

  const { dragHandleRef, dragPreviewRef, dropTargetRef, isDraggingOtherGroup, isOver } = (() => {
    // If draggable charts grid layout is enabled, don't use local drag and drop
    // but rely on the visibility provided by props instead
    if (shouldEnableDraggableChartsGridLayout()) {
      return {
        dragHandleRef: undefined,
        dragPreviewRef: undefined,
        dropTargetRef: undefined,
        isDraggingOtherGroup: false,
        isOver: false,
      };
    }
    // We can safely disable the eslint rule here because flag evaluation is stable
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useDragAndDropElement({
      dragGroupKey,
      dragKey: uuid || '',
      onDrop: onReorderWith,
    });
  })();

  const onMoveUp = useCallback(
    () => onReorderWith(uuid || '', previousChartUuid || ''),
    [onReorderWith, uuid, previousChartUuid],
  );
  const onMoveDown = useCallback(
    () => onReorderWith(uuid || '', nextChartUuid || ''),
    [onReorderWith, uuid, nextChartUuid],
  );

  return (
    <div
      css={{
        // Either use provided height or default to 360
        height: shouldEnableDraggableChartsGridLayout() ? height ?? 360 : 360,
        overflow: 'hidden',
        display: 'grid',
        gridTemplateRows: 'auto 1fr',
        backgroundColor: theme.colors.backgroundPrimary,
        padding: 12,
        // have a slightly smaller padding when the enableDeepLearningUI
        // flag is on to accomodate the legend in the charts
        paddingBottom: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        transition: 'opacity 0.12s',
        position: 'relative',
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
          display: 'flex',
          overflow: 'hidden',
        }}
      >
        <div
          ref={dragHandleRef}
          data-testid="experiment-view-compare-runs-card-drag-handle"
          css={{
            marginTop: theme.spacing.xs,
            marginRight: theme.spacing.sm,
            cursor: 'grab',
          }}
          className={DRAGGABLE_CARD_HANDLE_CLASS}
        >
          <DragIcon />
        </div>
        <div css={{ overflow: 'hidden', flex: 1, flexShrink: 1 }}>
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
          {tooltip && <LegacyInfoTooltip css={{ verticalAlign: 'middle' }} title={tooltip} />}
        </div>
        {isRefreshing && (
          <div
            css={{
              width: theme.general.heightSm,
              height: theme.general.heightSm,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Spinner />
          </div>
        )}
        {toggles && (
          <div
            css={{
              display: 'flex',
              padding: `0px ${theme.spacing.lg}px`,
              gap: theme.spacing.md,
              alignItems: 'flex-start',
            }}
          >
            {toggles.map((toggle) => {
              return (
                <Switch
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_262"
                  key={toggle.toggleLabel}
                  checked={toggle.currentToggle}
                  onChange={toggle.setToggle}
                  label={toggle.toggleLabel}
                />
              );
            })}
          </div>
        )}
        <Button
          componentId="fullscreen_button_chartcard"
          icon={<FullscreenIcon />}
          onClick={toggleFullScreenChart}
          disabled={!toggleFullScreenChart}
        />
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
            <DropdownMenu.Item
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_288"
              onClick={onEdit}
              data-testid="experiment-view-compare-runs-card-edit"
            >
              Configure
            </DropdownMenu.Item>
            <DropdownMenu.Item
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_291"
              onClick={onDelete}
              data-testid="experiment-view-compare-runs-card-delete"
            >
              Delete
            </DropdownMenu.Item>
            {supportedDownloadFormats.length > 0 && onClickDownload && (
              <>
                <DropdownMenu.Separator />
                {supportedDownloadFormats.includes('csv') && (
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_298"
                    onClick={() => onClickDownload('csv')}
                  >
                    <FormattedMessage
                      defaultMessage="Export as CSV"
                      description="Experiment page > compare runs tab > chart header > export CSV data option"
                    />
                  </DropdownMenu.Item>
                )}
                {supportedDownloadFormats.includes('svg') && (
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_316"
                    onClick={() => onClickDownload('svg')}
                  >
                    <FormattedMessage
                      defaultMessage="Download as SVG"
                      description="Experiment page > compare runs tab > chart header > download as SVG option"
                    />
                  </DropdownMenu.Item>
                )}
                {supportedDownloadFormats.includes('png') && (
                  <DropdownMenu.Item
                    componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_324"
                    onClick={() => onClickDownload('png')}
                  >
                    <FormattedMessage
                      defaultMessage="Download as PNG"
                      description="Experiment page > compare runs tab > chart header > download as PNG option"
                    />
                  </DropdownMenu.Item>
                )}
              </>
            )}
            <DropdownMenu.Separator />
            <DropdownMenu.Item
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_334"
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
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_344"
              disabled={!canMoveDown}
              onClick={onMoveDown}
              data-testid="experiment-view-compare-runs-move-down"
            >
              <FormattedMessage
                defaultMessage="Move down"
                description="Experiment page > compare runs tab > chart header > move down option"
              />
            </DropdownMenu.Item>
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

export const RunsChartCardWrapper = memo(RunsChartCardWrapperRaw);
