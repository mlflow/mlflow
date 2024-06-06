import { ImageIcon } from '@databricks/design-system';
import { GenericSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import { ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import { useState } from 'react';
import { Typography } from '@databricks/design-system';
import { ImagePreviewGroup, Image } from '../../../../../shared/building_blocks/Image';

export const MAX_IMAGE_SIZE = 225;
export const MIN_IMAGE_SIZE = 120;
export const IMAGE_GAP_SIZE = 10;

export const getImageSize = (numImages: number, width: number) => {
  // Scale image size based on number of images
  const maxImagesPerRow = Math.floor(width / MIN_IMAGE_SIZE);
  if (numImages < maxImagesPerRow) {
    return Math.min(width / numImages - IMAGE_GAP_SIZE, MAX_IMAGE_SIZE);
  }
  return width / maxImagesPerRow - IMAGE_GAP_SIZE;
};

type ImagePlotProps = {
  imageUrl: string;
  compressedImageUrl: string;
  imageSize?: number;
  maxImageSize?: number;
};

export const ImagePlot = ({ imageUrl, compressedImageUrl, imageSize, maxImageSize }: ImagePlotProps) => {
  const [previewVisible, setPreviewVisible] = useState(false);
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ width: imageSize, height: imageSize || '100%' }}>
      <div css={{ height: imageSize || '100%' }}>
        {compressedImageUrl === undefined ? (
          <GenericSkeleton label="Loading..." css={{ height: imageSize, width: imageSize }} />
        ) : (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: imageSize || '100%',
              height: imageSize || '100%',
              maxWidth: maxImageSize,
              maxHeight: maxImageSize,
              backgroundColor: theme.colors.backgroundSecondary,
              '& .ant-image': {
                height: '100%',
                display: 'flex',
                alignItems: 'center',
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
          height: imageSize,
          backgroundColor: theme.colors.backgroundSecondary,
          padding: theme.spacing.md,
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
