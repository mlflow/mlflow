import { LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';
import { useMemo } from 'react';
import { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import { RunsChartsRunData } from '../RunsCharts.common';
import { EmptyImageGridPlot, IMAGE_GAP_SIZE, ImagePlotWithHistory, getImageSize } from './ImageGridPlot.common';
import { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';

export const ImageGridSingleKeyPlot = ({
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

  const displayRuns = previewData.filter((run: RunsChartsRunData) => {
    const imageMetadata = run.images[cardConfig.imageKeys[0]];
    return imageMetadata && Object.keys(imageMetadata).length > 0;
  });

  const imageSize = useMemo(() => {
    return getImageSize(displayRuns.length, width);
  }, [displayRuns, width]);

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }
  return (
    <div css={{ display: 'flex', justifyContent: 'flex-start', flexWrap: 'wrap' }}>
      {displayRuns.map((run: RunsChartsRunData) => {
        // There is exactly one key in this plot
        const imageMetadataByStep = Object.values(run.images[cardConfig.imageKeys[0]]).reduce(
          (acc, metadata: ImageEntity) => {
            if (metadata.step !== undefined) {
              acc[metadata.step] = metadata;
            }
            return acc;
          },
          {} as Record<number, ImageEntity>,
        );
        return (
          <div key={run.uuid} css={{ padding: `${IMAGE_GAP_SIZE / 2}px` }}>
            <LegacyTooltip title={run.displayName}>
              <div
                css={{
                  width: imageSize,
                  height: theme.typography.lineHeightMd,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: theme.spacing.sm,
                }}
              >
                <RunColorPill color={run.color} />
                {run.displayName}
              </div>
            </LegacyTooltip>
            <ImagePlotWithHistory
              key={run.uuid}
              step={cardConfig.step}
              metadataByStep={imageMetadataByStep}
              imageSize={imageSize}
              runUuid={run.uuid}
            />
          </div>
        );
      })}
    </div>
  );
};
