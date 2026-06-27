import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LegacySkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { RunsChartsRunData } from '../RunsCharts.common';
import type { RunsChartCardFullScreenProps } from './ChartCard.common';
import { type RunsChartCardReorderProps, RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';
import type { RunsChartsCardConfig, RunsChartsVideoCardConfig } from '../../runs-charts.types';
import { getArtifactBlob, getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import { getBasename } from '@mlflow/mlflow/src/common/utils/FileUtils';
import { MlflowService } from '@mlflow/mlflow/src/experiment-tracking/sdk/MlflowService';
import {
  isVideoArtifactPath,
  sortVideoArtifacts,
  VIDEO_ARTIFACT_DIRECTORIES,
} from '@mlflow/mlflow/src/experiment-tracking/utils/VideoUtils';
import { extractStepNumber } from '@mlflow/mlflow/src/experiment-tracking/utils/VideoUtils';
import { LineSmoothSlider } from '@mlflow/mlflow/src/experiment-tracking/components/LineSmoothSlider';
import { FormattedMessage } from 'react-intl';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';
import type { RunsGroupByConfig } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/utils/experimentPage.group-row-utils';

const VIDEO_CHART_CARD_HEIGHT = 500;

export interface RunsChartsVideoChartCardProps extends RunsChartCardReorderProps, RunsChartCardFullScreenProps {
  config: RunsChartsVideoCardConfig;
  chartRunData: RunsChartsRunData[];
  onDelete: () => void;
  onEdit: () => void;
  groupBy: RunsGroupByConfig | null;
}

interface RunVideoInfo {
  videos: string[];
  loading: boolean;
}

/**
 * Fetch video artifacts for a single run
 */
async function fetchVideosForRun(runUuid: string): Promise<string[]> {
  const videoPaths: string[] = [];
  const scannedDirs = new Set<string>();

  const listDir = async (path?: string): Promise<{ path: string; is_dir: boolean }[]> => {
    const dirKey = path || '';
    if (scannedDirs.has(dirKey)) return [];
    scannedDirs.add(dirKey);
    try {
      const response = await MlflowService.listArtifacts({
        run_uuid: runUuid,
        ...(path && { path }),
      });
      return response.files || [];
    } catch {
      return [];
    }
  };

  const rootFiles = await listDir();
  for (const file of rootFiles) {
    if (!file.is_dir && isVideoArtifactPath(file.path)) {
      videoPaths.push(file.path);
    }
  }

  const dirsToScan = [...VIDEO_ARTIFACT_DIRECTORIES];
  for (const file of rootFiles) {
    if (file.is_dir) {
      const dirName = file.path.split('/').pop() || file.path;
      if (VIDEO_ARTIFACT_DIRECTORIES.some((vd) => vd === dirName || vd.startsWith(dirName + '/'))) {
        dirsToScan.push(file.path);
      }
    }
  }

  const dirResults = await Promise.all([...new Set(dirsToScan)].map((dir) => listDir(dir)));
  for (const files of dirResults) {
    for (const file of files) {
      if (!file.is_dir && isVideoArtifactPath(file.path)) {
        videoPaths.push(file.path);
      }
    }
  }

  return sortVideoArtifacts([...new Set(videoPaths)]);
}

/**
 * Single video cell for one run
 */
const VideoCell = ({
  runUuid,
  videoPath,
  displayName,
  color,
}: {
  runUuid: string;
  videoPath: string | undefined;
  displayName: string;
  color: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const [blobUrl, setBlobUrl] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const blobUrlRef = useRef<string | undefined>();

  useEffect(() => {
    if (!videoPath || !runUuid) {
      setBlobUrl(undefined);
      return;
    }

    setLoading(true);
    const artifactUrl = getArtifactLocationUrl(videoPath, runUuid);

    getArtifactBlob(artifactUrl)
      .then((blob: Blob) => {
        if (blobUrlRef.current) {
          URL.revokeObjectURL(blobUrlRef.current);
        }
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;
        setBlobUrl(url);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });

    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = undefined;
      }
    };
  }, [videoPath, runUuid]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
        minWidth: 0,
      }}
    >
      {/* Run header */}
      <div
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          overflow: 'hidden',
        }}
      >
        <RunColorPill color={color} />
        <Typography.Text
          ellipsis
          css={{ fontSize: theme.typography.fontSizeSm, flex: 1, minWidth: 0 }}
          title={displayName}
        >
          {displayName}
        </Typography.Text>
      </div>
      {/* Video area */}
      <div
        css={{
          backgroundColor: '#000',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: 160,
          aspectRatio: '4/3',
        }}
      >
        {loading && <LegacySkeleton active paragraph={{ rows: 1 }} />}
        {!videoPath && !loading && (
          <Typography.Text css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="No video" description="Video chart card > No video label" />
          </Typography.Text>
        )}
        {blobUrl && !loading && (
          <video
            key={videoPath}
            src={blobUrl}
            controls
            autoPlay
            muted
            loop
            preload="auto"
            aria-label="video"
            css={{
              maxWidth: '100%',
              maxHeight: '100%',
              display: 'block',
            }}
          >
            <track kind="captions" srcLang="en" src="" default />
          </video>
        )}
      </div>
    </div>
  );
};

export const RunsChartsVideoChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  fullScreen,
  setFullScreenChart,
  ...reorderProps
}: RunsChartsVideoChartCardProps) => {
  const { theme } = useDesignSystemTheme();
  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();

  const visibleRuns = useMemo(() => chartRunData.filter(({ hidden }) => !hidden).reverse(), [chartRunData]);

  // Fetch video artifacts for all visible runs
  const [runVideoMap, setRunVideoMap] = useState<Record<string, RunVideoInfo>>({});
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const runUuids = visibleRuns.map((r) => r.uuid);

    setRunVideoMap((prev) => {
      const next = { ...prev };
      for (const uuid of runUuids) {
        if (!next[uuid]) {
          next[uuid] = { videos: [], loading: true };
        }
      }
      return next;
    });

    Promise.all(
      runUuids.map(async (uuid) => {
        const videos = await fetchVideosForRun(uuid);
        return { uuid, videos };
      }),
    ).then((results) => {
      if (cancelled) return;
      setRunVideoMap((prev) => {
        const next = { ...prev };
        for (const { uuid, videos } of results) {
          next[uuid] = { videos, loading: false };
        }
        return next;
      });
      setInitialized(true);
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleRuns.map((r) => r.uuid).join(',')]);

  // Collect all unique steps across all runs
  const { allSteps, stepToVideoByRun } = useMemo(() => {
    const stepsSet = new Set<number>();
    const mapping: Record<string, Map<number, string>> = {};

    for (const run of visibleRuns) {
      const data = runVideoMap[run.uuid];
      if (!data || data.loading) continue;

      const videoPaths = data.videos;
      const stepMap = new Map<number, string>();

      for (const path of videoPaths) {
        const step = extractStepNumber(getBasename(path));
        if (step !== null && !stepMap.has(step)) {
          stepMap.set(step, path);
        }
      }

      if (stepMap.size === 0) {
        videoPaths.forEach((path, idx) => {
          stepMap.set(idx, path);
        });
      }

      for (const s of stepMap.keys()) {
        stepsSet.add(s);
      }
      mapping[run.uuid] = stepMap;
    }

    const sortedSteps = Array.from(stepsSet).sort((a, b) => a - b);
    return { allSteps: sortedSteps, stepToVideoByRun: mapping };
  }, [visibleRuns, runVideoMap]);

  // Local slider state for smooth dragging
  const [tmpStep, setTmpStep] = useState(config.step);

  useEffect(() => {
    setTmpStep(config.step);
  }, [config.step]);

  const updateStep = useCallback(
    (newStep: number) => {
      if (config.step === newStep) return;
      confirmChartCardConfiguration({ ...config, step: newStep } as RunsChartsVideoCardConfig);
    },
    [config, confirmChartCardConfiguration],
  );

  // Initialize step to first available if current is invalid
  useEffect(() => {
    if (allSteps.length > 0 && !allSteps.includes(config.step)) {
      updateStep(allSteps[0]);
    }
  }, [allSteps, config.step, updateStep]);

  const stepMarks = useMemo(() => {
    const marks: Record<number, string> = {};
    for (const step of allSteps) {
      marks[step] = String(step);
    }
    return marks;
  }, [allSteps]);

  const minStep = allSteps.length > 0 ? allSteps[0] : 0;
  const maxStep = allSteps.length > 0 ? allSteps[allSteps.length - 1] : 0;

  const getVideoPathForRunAtStep = useCallback(
    (runUuid: string, step: number): string | undefined => {
      const stepMap = stepToVideoByRun[runUuid];
      if (!stepMap) return undefined;
      if (stepMap.has(step)) return stepMap.get(step);
      let nearest: number | undefined;
      for (const s of stepMap.keys()) {
        if (s <= step && (nearest === undefined || s > nearest)) {
          nearest = s;
        }
      }
      return nearest !== undefined ? stepMap.get(nearest) : undefined;
    },
    [stepToVideoByRun],
  );

  const hasAnyVideos = Object.values(runVideoMap).some((d) => d.videos.length > 0);

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: 'Videos',
      subtitle: null,
    });
  };

  const numRuns = visibleRuns.length;
  const gridCols = Math.min(numRuns, 4);

  const chartBody = (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: fullScreen ? '100%' : undefined,
        width: '100%',
        overflow: 'hidden',
        marginTop: theme.spacing.sm,
        gap: theme.spacing.sm,
      }}
    >
      {!initialized && (
        <div css={{ padding: theme.spacing.md }}>
          <LegacySkeleton active paragraph={{ rows: 3 }} />
        </div>
      )}
      {initialized && !hasAnyVideos && (
        <div css={{ padding: theme.spacing.md, textAlign: 'center' }}>
          <Typography.Text css={{ color: theme.colors.textSecondary }}>
            <FormattedMessage
              defaultMessage="No video artifacts found"
              description="Video chart card > No videos message"
            />
          </Typography.Text>
        </div>
      )}
      {initialized && hasAnyVideos && (
        <>
          <div
            css={{
              flex: 1,
              overflow: 'auto',
              display: 'grid',
              gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
              gap: theme.spacing.sm,
              padding: `0 ${theme.spacing.sm}px`,
            }}
          >
            {visibleRuns.map((run) => {
              const videoPath = getVideoPathForRunAtStep(run.uuid, tmpStep);
              return (
                <VideoCell
                  key={run.uuid}
                  runUuid={run.uuid}
                  videoPath={videoPath}
                  displayName={run.displayName}
                  color={run.color}
                />
              );
            })}
          </div>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
              padding: `0 ${theme.spacing.md}px ${theme.spacing.sm}px`,
            }}
          >
            <Typography.Text
              css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary, whiteSpace: 'nowrap' }}
            >
              <FormattedMessage
                defaultMessage="Step: {step}"
                description="Video chart card > Step label"
                values={{ step: tmpStep }}
              />
            </Typography.Text>
            <div css={{ flex: 1 }}>
              <LineSmoothSlider
                value={tmpStep}
                onChange={setTmpStep}
                onAfterChange={updateStep}
                max={maxStep}
                min={minStep}
                marks={stepMarks}
                disabled={allSteps.length <= 1}
                css={{
                  '&[data-orientation="horizontal"]': { width: 'auto' },
                }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={config.displayName || 'Videos'}
      subtitle={hasAnyVideos ? `${allSteps.length} steps, ${numRuns} runs` : undefined}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      toggleFullScreenChart={toggleFullScreenChart}
      height={VIDEO_CHART_CARD_HEIGHT}
      {...reorderProps}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};
