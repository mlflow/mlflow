import {
  useDesignSystemTheme,
  TableRow,
  TableHeader,
  TableCell,
  Table,
  LegacyTooltip,
  Typography,
} from '@databricks/design-system';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';
import { useMemo } from 'react';
import { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { EmptyImageGridPlot, getImageSize, ImagePlotWithHistory } from './ImageGridPlot.common';
import { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { FormattedMessage } from 'react-intl';

export const ImageGridMultipleKeyPlot = ({
  previewData,
  cardConfig,
  width,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy?: string;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
  width: number;
}) => {
  const { theme } = useDesignSystemTheme();

  const displayRuns = previewData.filter((run: RunsChartsRunData) => Object.keys(run.images).length !== 0);

  const imageSize = useMemo(() => {
    return getImageSize(displayRuns.length, width);
  }, [displayRuns, width]);

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }
  return (
    <div css={{ height: '100%', width: '100%' }}>
      <Table grid scrollable>
        <TableRow isHeader>
          <TableHeader css={{ minWidth: imageSize + theme.spacing.md }}>
            <FormattedMessage
              defaultMessage="images"
              description="Experiment tracking > runs charts > charts > image grid multiple key > table header text"
            />
          </TableHeader>
          {displayRuns.map((run: RunsChartsRunData) => {
            return (
              <TableHeader key={run.uuid} css={{ minWidth: imageSize + theme.spacing.md }}>
                <LegacyTooltip title={run.displayName}>
                  <div
                    css={{
                      height: theme.typography.lineHeightMd,
                      whiteSpace: 'nowrap',
                      display: 'inline-flex',
                      alignItems: 'center',
                      margin: 'auto',
                      gap: theme.spacing.sm,
                    }}
                  >
                    <RunColorPill color={run.color} />
                    {run.displayName}
                  </div>
                </LegacyTooltip>
              </TableHeader>
            );
          })}
        </TableRow>
        {cardConfig.imageKeys.map((imageKey) => {
          return (
            <TableRow key={imageKey}>
              <TableCell css={{ minWidth: imageSize + theme.spacing.md }}>
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
                    <TableCell key={run.uuid} css={{ minWidth: imageSize + theme.spacing.md }}>
                      <ImagePlotWithHistory
                        metadataByStep={metadataByStep}
                        imageSize={imageSize}
                        step={cardConfig.step}
                        runUuid={run.uuid}
                      />
                    </TableCell>
                  );
                }
                return <TableCell key={run.uuid} css={{ minWidth: imageSize + theme.spacing.md }} />;
              })}
            </TableRow>
          );
        })}
      </Table>
    </div>
  );
};
