import React, { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { Typography, useDesignSystemTheme, LegacySkeleton } from '@databricks/design-system';
import { getArtifactBlob, getArtifactLocationUrl } from '../../../../common/utils/ArtifactUtils';
import { getBasename } from '../../../../common/utils/FileUtils';
import { useRunVideoArtifacts } from '../hooks/useRunVideoArtifacts';
import { extractStepNumber } from '../../../utils/VideoUtils';
import { LineSmoothSlider } from '../../LineSmoothSlider';
import { FormattedMessage } from 'react-intl';

interface RunViewVideosSectionProps {
  runUuid: string;
}

/**
 * W&B-style video panel with a step slider for scrubbing through video artifacts.
 * Shows one video at a time with a slider to navigate between different steps/episodes.
 * Renders in the Model Metrics Charts tab.
 */
export const RunViewVideosSection = ({ runUuid }: RunViewVideosSectionProps) => {
  const { videos, loading } = useRunVideoArtifacts(runUuid);
  const { theme } = useDesignSystemTheme();
  const [currentIndex, setCurrentIndex] = useState(0);

  // Group videos by their "key" (directory path prefix), W&B style
  const videoGroups = useMemo(() => {
    if (videos.length === 0) return [];

    // Group by directory prefix
    const groups = new Map<string, string[]>();
    for (const path of videos) {
      const parts = path.split('/');
      const key = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(path);
    }
    return Array.from(groups.entries()).map(([key, paths]) => ({
      key: key || 'videos',
      paths,
    }));
  }, [videos]);

  // For the currently selected group, build step marks for the slider
  const [selectedGroupIndex, setSelectedGroupIndex] = useState(0);
  const selectedGroup = videoGroups[selectedGroupIndex] ?? videoGroups[0];

  const { stepMarks, minStep, maxStep } = useMemo(() => {
    if (!selectedGroup) return { stepMarks: {} as Record<number, string>, minStep: 0, maxStep: 0 };

    const marks: Record<number, string> = {};
    selectedGroup.paths.forEach((path, index) => {
      const step = extractStepNumber(getBasename(path));
      const markValue = step !== null ? step : index;
      marks[markValue] = getBasename(path);
    });

    const markKeys = Object.keys(marks).map(Number);
    return {
      stepMarks: marks,
      minStep: Math.min(...markKeys),
      maxStep: Math.max(...markKeys),
    };
  }, [selectedGroup]);

  // Map slider value back to video index
  const sliderValueToIndex = useMemo(() => {
    if (!selectedGroup) return new Map<number, number>();
    const map = new Map<number, number>();
    selectedGroup.paths.forEach((path, index) => {
      const step = extractStepNumber(getBasename(path));
      map.set(step !== null ? step : index, index);
    });
    return map;
  }, [selectedGroup]);

  const [sliderValue, setSliderValue] = useState(minStep);

  const handleSliderChange = useCallback(
    (value: number) => {
      setSliderValue(value);
      const index = sliderValueToIndex.get(value);
      if (index !== undefined) {
        setCurrentIndex(index);
      }
    },
    [sliderValueToIndex],
  );

  // Fetch the video blob and create an object URL (required for auth headers)
  const [videoBlobUrl, setVideoBlobUrl] = useState<string | undefined>();
  const [videoLoading, setVideoLoading] = useState(false);
  const blobUrlRef = useRef<string | undefined>();

  const currentVideoPath = useMemo(
    () => selectedGroup?.paths[currentIndex] ?? videos[0],
    [selectedGroup, currentIndex, videos],
  );

  useEffect(() => {
    if (!currentVideoPath || !runUuid) return;

    setVideoLoading(true);
    const artifactUrl = getArtifactLocationUrl(currentVideoPath, runUuid);

    getArtifactBlob(artifactUrl).then((blob: Blob) => {
      // Revoke previous blob URL
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      blobUrlRef.current = url;
      setVideoBlobUrl(url);
      setVideoLoading(false);
    }).catch(() => {
      setVideoLoading(false);
    });

    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = undefined;
      }
    };
  }, [currentVideoPath, runUuid]);

  if (loading) {
    return <LegacySkeleton active paragraph={{ rows: 3 }} />;
  }

  if (videos.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundPrimary,
        overflow: 'hidden',
        marginBottom: theme.spacing.lg,
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
          <FormattedMessage defaultMessage="Videos" description="Run page > Charts tab > Videos panel title" />
          <Typography.Hint css={{ marginLeft: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
            ({videos.length})
          </Typography.Hint>
        </Typography.Title>
        {/* Group selector if multiple groups */}
        {videoGroups.length > 1 && (
          <select
            value={selectedGroupIndex}
            onChange={(e) => {
              setSelectedGroupIndex(Number(e.target.value));
              setCurrentIndex(0);
              setSliderValue(minStep);
            }}
            css={{
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              borderRadius: theme.borders.borderRadiusMd,
              border: `1px solid ${theme.colors.border}`,
              backgroundColor: theme.colors.backgroundPrimary,
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {videoGroups.map((group, idx) => (
              <option key={group.key} value={idx}>
                {group.key}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Video player */}
      <div
        css={{
          backgroundColor: '#000',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: 300,
        }}
      >
        {videoLoading && <LegacySkeleton active paragraph={{ rows: 2 }} />}
        {videoBlobUrl && !videoLoading && (
          <video
            key={currentVideoPath}
            src={videoBlobUrl}
            controls
            autoPlay
            preload="auto"
            aria-label="video"
            css={{
              maxWidth: '100%',
              maxHeight: '480px',
              display: 'block',
            }}
          >
            <track kind="captions" srcLang="en" src="" default />
          </video>
        )}
      </div>

      {/* Step slider + label */}
      <div
        css={{
          padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.md}px`,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.xs,
        }}
      >
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Typography.Text css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}>
            {getBasename(currentVideoPath)}
          </Typography.Text>
          <Typography.Text css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}>
            <FormattedMessage
              defaultMessage="Step: {step}"
              description="Run page > Charts tab > Videos panel > Step label"
              values={{ step: sliderValue }}
            />
          </Typography.Text>
        </div>
        <LineSmoothSlider
          value={sliderValue}
          onChange={handleSliderChange}
          onAfterChange={handleSliderChange}
          max={maxStep}
          min={minStep}
          marks={stepMarks}
          disabled={Object.keys(stepMarks).length <= 1}
        />
      </div>
    </div>
  );
};
