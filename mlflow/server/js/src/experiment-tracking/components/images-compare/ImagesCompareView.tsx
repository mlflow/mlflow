import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxTrigger,
  Empty,
  ImageIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { FormattedMessage } from 'react-intl';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import type { ReduxState } from '@mlflow/mlflow/src/redux-types';
import type { ImageEntity } from '../../types';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { LOG_IMAGE_TAG_INDICATOR, NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE } from '../../constants';
import { usePopulateImagesByRunUuid } from '../experiment-page/hooks/usePopulateImagesByRunUuid';
import { useGetExperimentRunColor } from '../experiment-page/hooks/useExperimentRunColor';
import { useImageSliderStepMarks } from '../runs-charts/hooks/useImageSliderStepMarks';
import { ImageGridRunHeader, ImagePlotWithHistory } from '../runs-charts/components/charts/ImageGridPlot.common';
import { LineSmoothSlider } from '../LineSmoothSlider';

const IMAGE_CARD_SIZE = 200;

interface ImagesCompareViewProps {
  comparedRuns: RunRowType[];
  autoRefreshEnabled?: boolean;
  disabled?: boolean;
}

export const ImagesCompareView = ({ comparedRuns, autoRefreshEnabled, disabled }: ImagesCompareViewProps) => {
  const { theme } = useDesignSystemTheme();

  if (disabled) {
    return (
      <div
        css={{
          flex: 1,
          backgroundColor: theme.colors.backgroundSecondary,
          height: '100%',
          minHeight: 400,
          width: '100%',
          borderTop: `1px solid ${theme.colors.border}`,
          borderLeft: `1px solid ${theme.colors.border}`,
          paddingTop: theme.spacing.lg,
          marginLeft: -1,
          zIndex: 1,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          '& > div': {
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          },
        }}
      >
        <Empty
          title={
            <FormattedMessage
              defaultMessage="Image comparison not available when grouping is enabled"
              description="Experiment page > images compare view > disabled due to run grouping > title"
            />
          }
          description={
            <FormattedMessage
              defaultMessage="Disable run grouping in order to access the image comparison view"
              description="Experiment page > images compare view > disabled due to run grouping > description"
            />
          }
          image={<div />}
        />
      </div>
    );
  }

  return <ImagesCompareViewImpl comparedRuns={comparedRuns} autoRefreshEnabled={autoRefreshEnabled} />;
};

const ImagesCompareViewImpl = ({ comparedRuns, autoRefreshEnabled }: Omit<ImagesCompareViewProps, 'disabled'>) => {
  const { theme } = useDesignSystemTheme();
  const getRunColor = useGetExperimentRunColor();

  const { paramsByRunUuid, tagsByRunUuid, imagesByRunUuid } = useSelector((state: ReduxState) => ({
    paramsByRunUuid: state.entities.paramsByRunUuid,
    tagsByRunUuid: state.entities.tagsByRunUuid,
    imagesByRunUuid: state.entities.imagesByRunUuid,
  }));

  const chartData: RunsChartsRunData[] = useMemo(
    () =>
      comparedRuns
        .filter((run) => run.runInfo && !run.hidden)
        .slice(0, NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE)
        .map((run) => ({
          uuid: run.runUuid,
          displayName: run.runInfo?.runName ?? run.runUuid,
          runInfo: run.runInfo,
          metrics: {},
          params: (paramsByRunUuid[run.runUuid] || {}) as Record<string, { key: string; value: string | number }>,
          tags: tagsByRunUuid[run.runUuid] || {},
          images: imagesByRunUuid[run.runUuid] || {},
          color: getRunColor(run.runUuid),
          pinned: run.pinned,
          pinnable: run.pinnable,
          metricsHistory: {},
          hidden: run.hidden,
        })),
    [comparedRuns, paramsByRunUuid, tagsByRunUuid, imagesByRunUuid, getRunColor],
  );

  // Fetch image metadata for runs that have logged images
  const runsWithImages = useMemo(() => chartData.filter((run) => run.tags[LOG_IMAGE_TAG_INDICATOR]), [chartData]);
  usePopulateImagesByRunUuid({
    runUuids: runsWithImages.map((r) => r.uuid),
    runUuidsIsActive: runsWithImages.map((r) => r.runInfo?.status === 'RUNNING'),
    enabled: true,
    autoRefreshEnabled,
  });

  const allImageKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const run of chartData) {
      for (const key of Object.keys(run.images)) {
        keys.add(key);
      }
    }
    return Array.from(keys).sort();
  }, [chartData]);

  const [selectedImageKeys, setSelectedImageKeys] = useState<string[]>([]);
  const hasInitializedKeys = useRef(false);

  useEffect(() => {
    if (allImageKeys.length > 0 && !hasInitializedKeys.current) {
      setSelectedImageKeys(allImageKeys);
      hasInitializedKeys.current = true;
    }
  }, [allImageKeys]);

  const handleImageKeyToggle = useCallback((key: string) => {
    setSelectedImageKeys((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  }, []);

  const { stepMarks, maxMark, minMark } = useImageSliderStepMarks({
    data: chartData,
    selectedImageKeys,
  });
  const [tmpStep, setTmpStep] = useState(0);

  const stepMarkLength = Object.keys(stepMarks).length;
  const safeMinMark = stepMarkLength > 0 ? minMark : 0;
  const safeMaxMark = stepMarkLength > 0 ? maxMark : 0;

  // Derive effective step: snap to the only available step when there's just one,
  // and fall back to minMark when the current step isn't in the marks
  const effectiveStep =
    stepMarkLength === 1 ? safeMinMark : stepMarkLength > 0 && !(tmpStep in stepMarks) ? safeMinMark : tmpStep;

  const displayRuns = useMemo(() => {
    if (selectedImageKeys.length === 0) return [];
    return chartData.filter((run) =>
      selectedImageKeys.some((key) => run.images[key] && Object.keys(run.images[key]).length > 0),
    );
  }, [chartData, selectedImageKeys]);

  const shouldDisplayImageLimitIndicator =
    comparedRuns.filter((run) => !run.hidden).length > NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE;

  // Loading state: runs have the logged-images tag but image metadata hasn't loaded yet
  if (runsWithImages.length > 0 && allImageKeys.length === 0) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
          width: '100%',
          borderLeft: `1px solid ${theme.colors.border}`,
        }}
      >
        <Spinner />
      </div>
    );
  }

  // Empty state: no runs have the logged-images tag
  if (runsWithImages.length === 0) {
    return (
      <div
        css={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 400,
          width: '100%',
          borderLeft: `1px solid ${theme.colors.border}`,
          '& > div': {
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          },
        }}
      >
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No logged images found. Use mlflow.log_image() to log images to your runs."
              description="Experiment page > images compare view > no images empty state"
            />
          }
          image={<ImageIcon />}
        />
      </div>
    );
  }

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        borderLeft: `1px solid ${theme.colors.border}`,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <DialogCombobox
          componentId="mlflow.images_compare.image_key_selector"
          multiSelect
          label={
            <FormattedMessage
              defaultMessage="Image keys"
              description="Experiment page > images compare view > image key selector label"
            />
          }
          value={selectedImageKeys}
        >
          <DialogComboboxTrigger showTagAfterValueCount={2} />
          <DialogComboboxContent>
            <DialogComboboxOptionList>
              {allImageKeys.map((key) => (
                <DialogComboboxOptionListCheckboxItem
                  key={key}
                  value={key}
                  checked={selectedImageKeys.includes(key)}
                  onChange={() => handleImageKeyToggle(key)}
                >
                  {key}
                </DialogComboboxOptionListCheckboxItem>
              ))}
            </DialogComboboxOptionList>
          </DialogComboboxContent>
        </DialogCombobox>

        {shouldDisplayImageLimitIndicator && (
          <Typography.Text color="warning" size="sm">
            <FormattedMessage
              defaultMessage="Displaying images from the top {maxRuns} runs"
              description="Experiment page > images compare view > run limit warning"
              values={{ maxRuns: NUM_RUNS_TO_SUPPORT_FOR_LOG_IMAGE }}
            />
          </Typography.Text>
        )}
      </div>

      <div
        css={{
          flex: 1,
          overflow: 'auto',
          padding: theme.spacing.md,
        }}
      >
        {displayRuns.length === 0 && selectedImageKeys.length > 0 ? (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              minHeight: 400,
              width: '100%',
              '& > div': {
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              },
            }}
          >
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="No runs have images for the selected keys"
                  description="Experiment page > images compare view > no runs for key empty state"
                />
              }
              image={<ImageIcon />}
            />
          </div>
        ) : (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
            {selectedImageKeys.map((imageKey) => (
              <div key={imageKey}>
                <Typography.Text bold css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                  {imageKey}
                </Typography.Text>
                <div
                  css={{
                    display: 'grid',
                    gridTemplateColumns: `repeat(auto-fill, minmax(${IMAGE_CARD_SIZE}px, 1fr))`,
                    gap: theme.spacing.sm,
                  }}
                >
                  {displayRuns.map((run) => {
                    const imageData = run.images[imageKey];
                    if (!imageData || Object.keys(imageData).length === 0) return null;
                    const imageMetadataByStep = Object.values(imageData as Record<string, ImageEntity>).reduce(
                      (acc, metadata) => {
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
                          display: 'flex',
                          flexDirection: 'column',
                          border: '1px solid transparent',
                          borderRadius: theme.borders.borderRadiusSm,
                          padding: theme.spacing.sm,
                          '&:hover': {
                            border: `1px solid ${theme.colors.border}`,
                            backgroundColor: theme.colors.tableBackgroundUnselectedHover,
                          },
                        }}
                      >
                        <ImageGridRunHeader
                          displayName={run.displayName}
                          color={run.color}
                          params={run.params as Record<string, { key: string; value: string | number }>}
                          maxParamsWidth={IMAGE_CARD_SIZE}
                        />
                        <div css={{ marginTop: 'auto' }}>
                          <ImagePlotWithHistory
                            step={effectiveStep}
                            metadataByStep={imageMetadataByStep}
                            runUuid={run.uuid}
                            imageSize={IMAGE_CARD_SIZE}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.md,
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          borderTop: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
        }}
      >
        <div css={{ flex: 1 }}>
          <LineSmoothSlider
            value={effectiveStep}
            onChange={setTmpStep}
            max={safeMaxMark}
            min={safeMinMark}
            marks={stepMarks}
            disabled={stepMarkLength <= 1}
            onAfterChange={setTmpStep}
            css={{
              '&[data-orientation="horizontal"]': { width: 'auto' },
            }}
          />
        </div>
      </div>
    </div>
  );
};
