import { ImageIcon, Spinner, Tooltip } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getArtifactBlob, getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import {
  fetchArtifactPreferPresigned,
  fetchRunArtifactPreferPresigned,
} from '@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils';
import { useState, useEffect, useCallback } from 'react';
import { Typography } from '@databricks/design-system';
import { ImagePreviewGroup, Image } from '../../../../../shared/building_blocks/Image';
import { RunColorPill } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/components/RunColorPill';

export const MIN_GRID_IMAGE_SIZE = 400;

// Params like 'name' and 'run_id' duplicate the run name so we skip them
const REDUNDANT_PARAM_KEYS = new Set(['name', 'run_name', 'run_id']);

export const formatRunParams = (params: Record<string, { key: string; value: string | number }>): string => {
  const entries = Object.values(params).filter((p) => !REDUNDANT_PARAM_KEYS.has(p.key));
  if (entries.length === 0) return '';
  return entries.map((p) => `${p.key}=${p.value}`).join(', ');
};

export const ImageGridRunHeader = ({
  displayName,
  color,
  params,
  maxParamsWidth,
  showParams = true,
}: {
  displayName: string;
  color?: string;
  params: Record<string, { key: string; value: string | number }>;
  maxParamsWidth?: number;
  showParams?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const paramsText = showParams ? formatRunParams(params) : '';

  return (
    <Tooltip
      content={paramsText ? `${displayName}\n${paramsText}` : displayName}
      componentId="mlflow.charts.image-plot.run-name-tooltip"
    >
      <div css={{ overflow: 'hidden' }}>
        <div
          css={{
            height: theme.typography.lineHeightMd,
            whiteSpace: 'nowrap',
            display: 'inline-flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          <RunColorPill color={color} />
          {displayName}
        </div>
        {paramsText && (
          <div
            css={{
              fontSize: theme.typography.fontSizeSm,
              color: theme.colors.textSecondary,
              whiteSpace: 'normal',
              lineHeight: 1.3,
              maxWidth: maxParamsWidth,
              overflow: 'hidden',
            }}
          >
            {paramsText}
          </div>
        )}
      </div>
    </Tooltip>
  );
};

type ImagePlotProps = {
  imageUrl?: string;
  compressedImageUrl?: string;
  imageError?: boolean;
  previewError?: boolean;
  imageSize?: number;
  maxImageSize?: number;
  onPreviewVisibleChange?: (visible: boolean) => void;
};

export const ImagePlot = ({
  imageUrl,
  compressedImageUrl,
  imageError = false,
  previewError = false,
  imageSize,
  maxImageSize,
  onPreviewVisibleChange,
}: ImagePlotProps) => {
  const [previewVisible, setPreviewVisible] = useState(false);
  const { theme } = useDesignSystemTheme();

  const [imageLoading, setImageLoading] = useState(true);
  const [imageLoadFailed, setImageLoadFailed] = useState(false);

  useEffect(() => {
    // Load the image in the memory (should reuse the same request) in order to get the loading state
    setImageLoading(true);
    setImageLoadFailed(false);
    if (!compressedImageUrl) return;
    const img = new window.Image();
    img.onload = () => setImageLoading(false);
    img.onerror = () => {
      setImageLoading(false);
      setImageLoadFailed(true);
    };
    img.src = compressedImageUrl;
    return () => {
      img.onload = null;
      img.onerror = null;
      img.src = '';
    };
  }, [compressedImageUrl]);

  useEffect(() => {
    if (previewError) setPreviewVisible(false);
  }, [previewError]);

  return (
    <div
      aria-busy={!imageError && !imageLoadFailed && (compressedImageUrl === undefined || imageLoading)}
      style={{
        width: imageSize || '100%',
        height: imageSize || '100%',
        maxWidth: maxImageSize,
        maxHeight: maxImageSize,
        aspectRatio: '1',
      }}
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'relative',
        '.rc-image': { cursor: 'pointer' },
      }}
    >
      {imageError || imageLoadFailed ? (
        <Typography.Text color="info">
          <ImageIcon />{' '}
          <FormattedMessage
            defaultMessage="Image unavailable"
            description="Image grid > error displayed in a cell when its image cannot be loaded"
          />
        </Typography.Text>
      ) : compressedImageUrl === undefined || imageLoading ? (
        <Spinner />
      ) : (
        <>
          <ImagePreviewGroup
            visible={previewVisible}
            onVisibleChange={(visible) => {
              setPreviewVisible(visible);
              onPreviewVisibleChange?.(visible);
            }}
          >
            <Image
              src={compressedImageUrl}
              preview={previewError ? false : { src: imageUrl ?? compressedImageUrl }}
              style={{ maxWidth: maxImageSize || '100%', maxHeight: maxImageSize || '100%' }}
            />
          </ImagePreviewGroup>
          {previewError && (
            <Typography.Text
              color="info"
              css={{
                position: 'absolute',
                bottom: 0,
                width: '100%',
                textAlign: 'center',
                whiteSpace: 'normal',
                backgroundColor: theme.colors.backgroundSecondary,
              }}
            >
              <FormattedMessage
                defaultMessage="Full-size image unavailable"
                description="Image grid > error displayed in a cell when its full-size preview cannot be loaded"
              />
            </Typography.Text>
          )}
        </>
      )}
    </div>
  );
};

/** Fetch the thumbnail immediately and the original only when its preview opens. */
export const ArtifactImagePlot = ({
  filepath,
  compressedFilepath,
  fetchImage,
  imageSize,
  maxImageSize,
}: {
  filepath: string;
  compressedFilepath: string;
  fetchImage: (path: string) => Promise<Blob>;
  imageSize?: number;
  maxImageSize?: number;
}) => {
  const [compressedImageUrl, setCompressedImageUrl] = useState<string>();
  const [imageUrl, setImageUrl] = useState<string>();
  const [previewRequested, setPreviewRequested] = useState(false);
  const [thumbnailError, setThumbnailError] = useState(false);
  const [previewError, setPreviewError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let url: string | undefined;
    setCompressedImageUrl(undefined);
    setThumbnailError(false);
    fetchImage(compressedFilepath)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setCompressedImageUrl(url);
      })
      .catch(() => {
        if (!cancelled) setThumbnailError(true);
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [compressedFilepath, fetchImage]);

  useEffect(() => {
    if (!previewRequested || filepath === compressedFilepath) return;
    let cancelled = false;
    let url: string | undefined;
    setImageUrl(undefined);
    setPreviewError(false);
    fetchImage(filepath)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setImageUrl(url);
      })
      .catch(() => {
        if (!cancelled) setPreviewError(true);
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [compressedFilepath, fetchImage, filepath, previewRequested]);

  return (
    <ImagePlot
      compressedImageUrl={compressedImageUrl}
      imageUrl={imageUrl ?? compressedImageUrl}
      imageError={thumbnailError}
      previewError={previewError}
      imageSize={imageSize}
      maxImageSize={maxImageSize}
      onPreviewVisibleChange={(visible) => {
        if (visible) setPreviewRequested(true);
      }}
    />
  );
};

export const RunArtifactImagePlot = ({
  runUuid,
  filepath,
  compressedFilepath,
  artifactRootUri,
  imageSize,
  maxImageSize,
}: {
  runUuid: string;
  filepath: string;
  compressedFilepath: string;
  artifactRootUri?: string;
  imageSize?: number;
  maxImageSize?: number;
}) => {
  const fetchImage = useCallback(
    (path: string) => {
      const legacyArtifactLocation = getArtifactLocationUrl(path, runUuid);
      return artifactRootUri
        ? fetchArtifactPreferPresigned({ runUuid, path, artifactRootUri }, legacyArtifactLocation, getArtifactBlob)
        : fetchRunArtifactPreferPresigned(runUuid, path, legacyArtifactLocation, getArtifactBlob);
    },
    [artifactRootUri, runUuid],
  );

  return (
    <ArtifactImagePlot
      key={JSON.stringify([runUuid, artifactRootUri, filepath, compressedFilepath])}
      filepath={filepath}
      compressedFilepath={compressedFilepath}
      fetchImage={fetchImage}
      imageSize={imageSize}
      maxImageSize={maxImageSize}
    />
  );
};

export const ImagePlotWithHistory = ({
  metadataByStep,
  imageSize,
  step,
  runUuid,
  artifactRootUri,
}: {
  metadataByStep: Record<number, ImageEntity>;
  imageSize?: number;
  step: number;
  runUuid: string;
  artifactRootUri?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  if (metadataByStep[step] === undefined) {
    return (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          width: imageSize,
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
          aspectRatio: '1',
        }}
      >
        <ImageIcon />
        <FormattedMessage
          defaultMessage="No image logged at this step"
          description="Experiment tracking > runs charts > charts > image plot with history > no image text"
        />
      </div>
    );
  }
  return (
    <RunArtifactImagePlot
      runUuid={runUuid}
      filepath={metadataByStep[step].filepath}
      compressedFilepath={metadataByStep[step].compressed_filepath}
      artifactRootUri={artifactRootUri}
      imageSize={imageSize}
    />
  );
};

export const EmptyImageGridPlot = () => {
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        width: '100%',
        fontSize: 16,
      }}
    >
      <Typography.Title css={{ marginTop: 16 }} color="secondary" level={3}>
        Compare logged images
      </Typography.Title>
      <Typography.Text css={{ marginBottom: 16 }} color="secondary">
        Use the image grid chart to compare logged images across runs.
      </Typography.Text>
    </div>
  );
};
