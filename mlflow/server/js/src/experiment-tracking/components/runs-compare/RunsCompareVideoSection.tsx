import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Typography, useDesignSystemTheme, LegacySkeleton } from '@databricks/design-system';
import { getArtifactBlob, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { getBasename } from '../../../common/utils/FileUtils';
import { MlflowService } from '../../sdk/MlflowService';
import { isVideoArtifactPath, sortVideoArtifacts, VIDEO_ARTIFACT_DIRECTORIES } from '../../utils/VideoUtils';
import { extractStepNumber } from '../../utils/VideoUtils';
import { LineSmoothSlider } from '../LineSmoothSlider';
import { FormattedMessage } from 'react-intl';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { RunColorPill } from '../experiment-page/components/RunColorPill';

interface RunsCompareVideoSectionProps {
  chartRunData: RunsChartsRunData[];
}

interface RunVideoInfo {
  videos: string[];
  loading: boolean;
}

/**
 * Single video cell for one run at one step
 */
const VideoCell = ({
  runUuid,
  videoPath,
  displayName,
  color,
  stepLabel,
}: {
  runUuid: string;
  videoPath: string | undefined;
  displayName: string;
  color: string;
  stepLabel: string;
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
        backgroundColor: theme.colors.backgroundPrimary,
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
          minHeight: 180,
          aspectRatio: '4/3',
        }}
      >
        {loading && <LegacySkeleton active paragraph={{ rows: 1 }} />}
        {!videoPath && !loading && (
          <Typography.Text css={{ color: theme.colors.textSecondary, fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="No video" description="Compare runs > Video grid > No video label" />
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
      {/* Step label */}
      <div
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          borderTop: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text css={{ fontSize: theme.typography.fontSizeXs, color: theme.colors.textSecondary }} ellipsis>
          {stepLabel}
        </Typography.Text>
      </div>
    </div>
  );
};

/**
 * Fetch video artifacts for a single run (standalone async function, not a hook)
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

  // Scan root
  const rootFiles = await listDir();
  for (const file of rootFiles) {
    if (!file.is_dir && isVideoArtifactPath(file.path)) {
      videoPaths.push(file.path);
    }
  }

  // Scan known video directories
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

  // Deduplicate and sort
  const unique = [...new Set(videoPaths)];
  return sortVideoArtifacts(unique);
}

/**
 * W&B-style video comparison grid for multiple runs.
 * Shows a grid of video players (one per run) with a shared step slider.
 * Automatically discovers video artifacts across all compared runs.
 */
export const RunsCompareVideoSection = ({ chartRunData }: RunsCompareVideoSectionProps) => {
  const { theme } = useDesignSystemTheme();

  // Only consider visible (non-hidden) runs
  const visibleRuns = useMemo(() => chartRunData.filter((r) => !r.hidden), [chartRunData]);

  // Store video data per run
  const [runVideoMap, setRunVideoMap] = useState<Record<string, RunVideoInfo>>({});
  const [initialized, setInitialized] = useState(false);

  // Fetch videos for all visible runs
  useEffect(() => {
    let cancelled = false;
    const runUuids = visibleRuns.map((r) => r.uuid);

    // Mark all as loading
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

      // If no step numbers found, use index as step
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

  // Slider state
  const [sliderValue, setSliderValue] = useState(0);

  // Update slider bounds when steps change
  useEffect(() => {
    if (allSteps.length > 0 && !allSteps.includes(sliderValue)) {
      setSliderValue(allSteps[0]);
    }
  }, [allSteps, sliderValue]);

  const stepMarks = useMemo(() => {
    const marks: Record<number, string> = {};
    for (const step of allSteps) {
      marks[step] = String(step);
    }
    return marks;
  }, [allSteps]);

  const minStep = allSteps.length > 0 ? allSteps[0] : 0;
  const maxStep = allSteps.length > 0 ? allSteps[allSteps.length - 1] : 0;

  // Find the nearest available step for each run
  const getVideoPathForRunAtStep = useCallback(
    (runUuid: string, step: number): string | undefined => {
      const stepMap = stepToVideoByRun[runUuid];
      if (!stepMap) return undefined;

      // Exact match
      if (stepMap.has(step)) return stepMap.get(step);

      // Find nearest step <= requested step
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

  const isLoading = !initialized;
  const hasAnyVideos = Object.values(runVideoMap).some((d) => d.videos.length > 0);

  // Don't render anything if no videos found after loading
  if (initialized && !hasAnyVideos) {
    return null;
  }

  // Determine grid columns based on number of runs
  const numRuns = visibleRuns.length;
  const gridCols = Math.min(numRuns, 4); // Max 4 columns like W&B

  return (
    <div
      css={{
        marginBottom: theme.spacing.lg,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundPrimary,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Typography.Title level={4} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="Videos" description="Compare runs > Video section title" />
          {hasAnyVideos && (
            <Typography.Hint css={{ marginLeft: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
              ({allSteps.length} steps, {numRuns} runs)
            </Typography.Hint>
          )}
        </Typography.Title>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div css={{ padding: theme.spacing.md }}>
          <LegacySkeleton active paragraph={{ rows: 3 }} />
        </div>
      )}

      {/* Video grid */}
      {hasAnyVideos && !isLoading && (
        <>
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: `repeat(${gridCols}, 1fr)`,
              gap: theme.spacing.sm,
              padding: theme.spacing.sm,
            }}
          >
            {visibleRuns.map((run) => {
              const videoPath = getVideoPathForRunAtStep(run.uuid, sliderValue);
              const stepLabel = videoPath ? getBasename(videoPath) : `Step ${sliderValue}`;
              return (
                <VideoCell
                  key={run.uuid}
                  runUuid={run.uuid}
                  videoPath={videoPath}
                  displayName={run.displayName}
                  color={run.color}
                  stepLabel={stepLabel}
                />
              );
            })}
          </div>

          {/* Shared step slider */}
          <div
            css={{
              padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.md}px`,
              borderTop: `1px solid ${theme.colors.border}`,
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.md,
            }}
          >
            <Typography.Text css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary, whiteSpace: 'nowrap' }}>
              <FormattedMessage
                defaultMessage="Step: {step}"
                description="Compare runs > Video section > Step label"
                values={{ step: sliderValue }}
              />
            </Typography.Text>
            <div css={{ flex: 1 }}>
              <LineSmoothSlider
                value={sliderValue}
                onChange={setSliderValue}
                onAfterChange={setSliderValue}
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
};
