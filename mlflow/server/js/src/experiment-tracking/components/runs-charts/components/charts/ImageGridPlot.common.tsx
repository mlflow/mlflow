import { ImageIcon, Spinner } from '@databricks/design-system';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import type { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { useState, useEffect } from 'react';
import { Typography } from '@databricks/design-system';
import { ImagePreviewGroup, Image } from '../../../../../shared/building_blocks/Image';

/**
 * Despite image size being dynamic, we want to set a minimum size for the grid images.
 */
export const MIN_GRID_IMAGE_SIZE = 200;

type ImagePlotProps = {
  imageUrl: string;
  compressedImageUrl: string;
  imageSize?: number;
  maxImageSize?: number;
};

export const ImagePlot = ({ imageUrl, compressedImageUrl, imageSize, maxImageSize }: ImagePlotProps) => {
  const [previewVisible, setPreviewVisible] = useState(false);
  const { theme } = useDesignSystemTheme();

  const [imageLoading, setImageLoading] = useState(true);

  useEffect(() => {
    // Load the image in the memory (should reuse the same request) in order to get the loading state
    setImageLoading(true);
    const img = new window.Image();
    img.onload = () => setImageLoading(false);
    img.onerror = () => setImageLoading(false);
    img.src = compressedImageUrl;
    return () => {
      img.src = '';
    };
  }, [compressedImageUrl]);

  return (
    <div css={{ width: imageSize || '100%', height: imageSize || '100%' }}>
      <div css={{ display: 'contents' }}>
        {compressedImageUrl === undefined || imageLoading ? (
          <div
            css={{
              width: '100%',
              backgroundColor: theme.colors.backgroundSecondary,
              display: 'flex',
              aspectRatio: '1',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <Spinner />
          </div>
        ) : (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: imageSize || '100%',
              aspectRatio: '1',
              maxWidth: maxImageSize,
              maxHeight: maxImageSize,
              backgroundColor: theme.colors.backgroundSecondary,
              '.rc-image': {
                cursor: 'pointer',
              },
            }}
          >
            <ImagePreviewGroup visible={previewVisible} onVisibleChange={setPreviewVisible}>
              <Image
                src={compressedImageUrl}
                preview={{ src: imageUrl }}
                style={{ maxWidth: maxImageSize || '100%', maxHeight: maxImageSize || '100%' }}
              />
            </ImagePreviewGroup>
          </div>
        )}
      </div>
    </div>
  );
};

export const ImagePlotWithHistory = ({
  metadataByStep,
  imageSize,
  step,
  runUuid,
}: {
  metadataByStep: Record<number, ImageEntity>;
  imageSize?: number;
  step: number;
  runUuid: string;
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
    <ImagePlot
      imageUrl={getArtifactLocationUrl(metadataByStep[step].filepath, runUuid)}
      compressedImageUrl={getArtifactLocationUrl(metadataByStep[step].compressed_filepath, runUuid)}
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
