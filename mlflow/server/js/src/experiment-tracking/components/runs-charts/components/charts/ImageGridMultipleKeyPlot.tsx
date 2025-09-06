import { useDesignSystemTheme, TableRow, TableHeader, TableCell, Table, Tooltip } from '@databricks/design-system';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';
import { useMemo } from 'react';
import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { EmptyImageGridPlot, ImagePlotWithHistory, MIN_GRID_IMAGE_SIZE } from './ImageGridPlot.common';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { FormattedMessage } from 'react-intl';

export const ImageGridMultipleKeyPlot = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy?: string;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const displayRuns = previewData.filter((run: RunsChartsRunData) => Object.keys(run.images).length !== 0);

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }
  return (
    <div css={{ height: '100%', width: '100%' }}>
      <Table grid scrollable>
        <TableRow isHeader>
          <TableHeader
            componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_44"
            css={{ minWidth: MIN_GRID_IMAGE_SIZE + theme.spacing.md }}
          >
            <FormattedMessage
              defaultMessage="images"
              description="Experiment tracking > runs charts > charts > image grid multiple key > table header text"
            />
          </TableHeader>
          {displayRuns.map((run: RunsChartsRunData) => {
            return (
              <TableHeader
                componentId="codegen_mlflow_app_src_experiment-tracking_components_runs-charts_components_charts_imagegridmultiplekeyplot.tsx_52"
                key={run.uuid}
                css={{ minWidth: MIN_GRID_IMAGE_SIZE + theme.spacing.md }}
              >
                <Tooltip content={run.displayName} componentId="mlflow.charts.image-plot.run-name-tooltip">
                  <div
                    css={{
                      height: theme.typography.lineHeightMd,
                      whiteSpace: 'nowrap',
                      display: 'inline-flex',
                      alignItems: 'center',
                      margin: 'auto',
                      gap: theme.spacing.sm,
                      fontWeight: 'normal',
                    }}
                  >
                    <RunColorPill color={run.color} />
                    {run.displayName}
                  </div>
                </Tooltip>
              </TableHeader>
            );
          })}
        </TableRow>
        {cardConfig.imageKeys.map((imageKey) => {
          return (
            <TableRow key={imageKey}>
              <TableCell css={{ minWidth: MIN_GRID_IMAGE_SIZE + theme.spacing.md }}>
                <div style={{ whiteSpace: 'normal' }}>{imageKey}</div>
              </TableCell>
              {displayRuns.map((run: RunsChartsRunData) => {
                if (run.images[imageKey] && Object.keys(run.images[imageKey]).length > 0) {
                  const metadataByStep = Object.values(run.images[imageKey]).reduce((acc, metadata) => {
                    if (metadata.step !== undefined) {
                      acc[metadata.step] = metadata;
                    }
                    return acc;
                  }, {} as Record<number, ImageEntity>);
                  return (
                    <TableCell
                      key={run.uuid}
                      css={{
                        minWidth: MIN_GRID_IMAGE_SIZE + theme.spacing.md,
                        '&:hover': {
                          backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                        },
                      }}
                    >
                      <ImagePlotWithHistory metadataByStep={metadataByStep} step={cardConfig.step} runUuid={run.uuid} />
                    </TableCell>
                  );
                }
                return <TableCell key={run.uuid} css={{ minWidth: MIN_GRID_IMAGE_SIZE + theme.spacing.md }} />;
              })}
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};
