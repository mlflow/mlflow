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
import type { Theme } from '@emotion/react';
import type { PropsWithChildren, ReactNode } from 'react';
import React, { memo, useCallback, forwardRef } from 'react';
import { FormattedMessage } from 'react-intl';
import { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartsCardConfig } from '../../runs-charts.types';
import type { ExperimentChartImageDownloadFileFormat } from '../../hooks/useChartImageDownloadHandler';
import { noop } from 'lodash';

export const DRAGGABLE_CARD_HANDLE_CLASS = 'mlflow-charts-drag-handle';
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
  canMoveToTop: boolean;
  canMoveToBottom: boolean;
  firstChartUuid?: string;
  lastChartUuid?: string;
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
  title: string | ReactNode;
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

/**
 * Wrapper components for all chart cards. Provides styles and adds
 * a dropdown menu with actions for configure and delete.
 */
const RunsChartCardWrapperRaw = ({
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
  canMoveToTop,
  canMoveToBottom,
  previousChartUuid,
  nextChartUuid,
  firstChartUuid,
  lastChartUuid,
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

  const onMoveUp = useCallback(
    () => onReorderWith(uuid || '', previousChartUuid || ''),
    [onReorderWith, uuid, previousChartUuid],
  );
  const onMoveDown = useCallback(
    () => onReorderWith(uuid || '', nextChartUuid || ''),
    [onReorderWith, uuid, nextChartUuid],
  );
  const onMoveToTop = useCallback(
    () => onReorderWith(uuid || '', firstChartUuid || ''),
    [onReorderWith, uuid, firstChartUuid],
  );
  const onMoveToBottom = useCallback(
    () => onReorderWith(uuid || '', lastChartUuid || ''),
    [onReorderWith, uuid, lastChartUuid],
  );

  const usingCustomTitle = React.isValidElement(title);

  return (
    <div
      css={{
        // Either use provided height or default to 360
        height: height ?? 360,
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
      data-testid="experiment-view-compare-runs-card"
    >
      <div
        css={{
          display: 'flex',
          overflow: 'hidden',
        }}
      >
        <div
          data-testid="experiment-view-compare-runs-card-drag-handle"
          css={{
            marginTop: usingCustomTitle ? theme.spacing.sm : theme.spacing.xs,
            marginRight: theme.spacing.sm,
            cursor: 'grab',
          }}
          className={DRAGGABLE_CARD_HANDLE_CLASS}
        >
          <DragIcon />
        </div>
        {usingCustomTitle ? (
          title
        ) : (
          <div css={{ overflow: 'hidden', flex: 1, flexShrink: 1 }}>
            <Typography.Title
              title={String(title)}
              level={4}
              withoutMargins
              css={{
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
        )}
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
              disabled={!canMoveToTop}
              onClick={onMoveToTop}
              data-testid="experiment-view-compare-runs-move-to-top"
            >
              <FormattedMessage
                defaultMessage="Move to top"
                description="Experiment page > compare runs tab > chart header > move to top option"
              />
            </DropdownMenu.Item>
            <DropdownMenu.Item
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_340"
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
            <DropdownMenu.Item
              componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_cards_chartcard.common.tsx_350"
              disabled={!canMoveToBottom}
              onClick={onMoveToBottom}
              data-testid="experiment-view-compare-runs-move-to-bottom"
            >
              <FormattedMessage
                defaultMessage="Move to bottom"
                description="Experiment page > compare runs tab > chart header > move to bottom option"
              />
            </DropdownMenu.Item>
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

export const RunsChartCardLoadingPlaceholder = forwardRef<
  HTMLDivElement,
  {
    className?: string;
    style?: React.CSSProperties;
  }
>(({ className, style }, ref) => (
  <div
    css={{ display: 'flex', height: '100%', justifyContent: 'center', alignItems: 'center' }}
    className={className}
    style={style}
    ref={ref}
  >
    <Spinner />
  </div>
));
