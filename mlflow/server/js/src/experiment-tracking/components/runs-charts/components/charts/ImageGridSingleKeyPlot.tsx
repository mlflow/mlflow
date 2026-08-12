import { useDesignSystemTheme } from '@databricks/design-system';
import type { RunsChartsImageCardConfig, RunsChartsCardConfig } from '../../runs-charts.types';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { EmptyImageGridPlot, ImagePlotWithHistory, ImageGridRunHeader } from './ImageGridPlot.common';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';

export const ImageGridSingleKeyPlot = ({
  previewData,
  cardConfig,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsImageCardConfig;
  groupBy?: string;
  setCardConfig?: (setter: (current: RunsChartsCardConfig) => RunsChartsImageCardConfig) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const displayRuns = previewData.filter((run: RunsChartsRunData) => {
    const imageMetadata = run.images[cardConfig.imageKeys[0]];
    return imageMetadata && Object.keys(imageMetadata).length > 0;
  });

  if (displayRuns.length === 0) {
    return <EmptyImageGridPlot />;
  }
  return (
    <div
      css={{
        display: 'flex',
        justifyContent: 'flex-start',
        flexWrap: 'wrap',
        gap: theme.spacing.xs,
        height: '100%',
        minHeight: 0,
      }}
    >
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
          <div
            key={run.uuid}
            css={{
              border: `1px solid transparent`,
              borderRadius: theme.borders.borderRadiusSm,
              padding: theme.spacing.sm,
              // Column so the run header keeps its natural height and the media
              // absorbs the remainder of the card.
              display: 'flex',
              flexDirection: 'column',
              maxHeight: '100%',
              minHeight: 0,
              '&:hover': {
                border: `1px solid ${theme.colors.border}`,
                backgroundColor: theme.colors.tableBackgroundUnselectedHover,
              },
            }}
          >
            <ImageGridRunHeader displayName={run.displayName} color={run.color} params={run.params} />
            <ImagePlotWithHistory
              key={run.uuid}
              step={cardConfig.step}
              metadataByStep={imageMetadataByStep}
              runUuid={run.uuid}
            />
          </div>
        );
      })}
    </div>
  );
};
