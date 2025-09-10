import { fulfilled } from '@mlflow/mlflow/src/common/utils/ActionUtils';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import type { ListImagesAction } from '@mlflow/mlflow/src/experiment-tracking/actions';
import { LIST_IMAGES_API } from '@mlflow/mlflow/src/experiment-tracking/actions';
import {
  IMAGE_COMPRESSED_FILE_EXTENSION,
  IMAGE_FILE_EXTENSION,
  MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import type { ArtifactFileInfo, ImageEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import type { AsyncFulfilledAction } from '@mlflow/mlflow/src/redux-types';

class ImagePathParseError extends Error {
  public filename: string;

  constructor(message: string, filename: string) {
    super(message);
    this.filename = filename;
    this.name = 'ImagePathParseError';
  }
}

const IMAGE_FILEPATH_DELIMITERS = ['%', '+'];

const parseImageFile = (filename: string) => {
  // Extract extension
  const extension = filename.split('.').pop();
  let fileKey = extension ? filename.slice(0, -(extension.length + 1)) : filename;

  const delimiter = IMAGE_FILEPATH_DELIMITERS.find((delimiter) => fileKey.includes(delimiter));
  if (delimiter === undefined) {
    throw new ImagePathParseError('Logged image path parse: incorrect filename format for image file', filename);
  }
  // The variables retrieved here are not reliable on OSS due to the usage of "%" as the separator.
  // Need to switch to a different separator on the backend to fully resolve the issue.
  const [serializedImageKey, stepLabel, stepString, timestampLabel, timestampString, _, compressed] =
    fileKey.split(delimiter);
  const isCompressed = fileKey.endsWith('compressed');

  if (stepLabel !== 'step' || timestampLabel !== 'timestamp') {
    throw new ImagePathParseError(
      'Logged image path parse: failed to parse step and timestamp from image filename',
      filename,
    );
  }

  const step = parseInt(stepString, 10);
  const timestamp = parseInt(timestampString, 10);
  const imageKey = serializedImageKey.replace(/#/g, '/');

  if (isCompressed) {
    fileKey = fileKey.slice(0, -('compressed'.length + 1));
  }
  return { imageKey, step, timestamp, fileKey, extension, isCompressed };
};

// TODO: add tests for this reducer
export const imagesByRunUuid = (
  state: Record<string, Record<string, Record<string, ImageEntity>>> = {},
  action: AsyncFulfilledAction<ListImagesAction>,
) => {
  switch (action.type) {
    case fulfilled(LIST_IMAGES_API): {
      if (!action.meta) {
        return state;
      }
      // Populate state with image keys
      const { runUuid } = action.meta;
      const { files } = action.payload;
      try {
        if (!files) {
          // There are no images for this run
          return {
            ...state,
            [runUuid]: {},
          };
        }
        // Filter images to only include directories
        const result = files.reduce((acc: Record<string, Record<string, ImageEntity>>, file: ArtifactFileInfo) => {
          if (!file.is_dir) {
            if (!file.path) {
              return acc;
            }
            const { imageKey, step, timestamp, fileKey, extension, isCompressed } = parseImageFile(
              file.path.slice((MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH + '/').length),
            );

            // Double check extension of image files
            if (extension === IMAGE_FILE_EXTENSION || extension === IMAGE_COMPRESSED_FILE_EXTENSION) {
              if (isCompressed) {
                acc[imageKey] = {
                  ...acc[imageKey],
                  [fileKey]: {
                    ...acc[imageKey]?.[fileKey],
                    compressed_filepath: file.path,
                  },
                };
              } else {
                // Set the step and timestamp when retrieving the uncompressed image file.
                acc[imageKey] = {
                  ...acc[imageKey],
                  [fileKey]: {
                    ...acc[imageKey]?.[fileKey],
                    filepath: file.path,
                    step: step,
                    timestamp: timestamp,
                  },
                };
              }
            }
          }
          return acc;
        }, {} as Record<string, Record<string, ImageEntity>>);
        return {
          ...state,
          [runUuid]: result,
        };
      } catch (e) {
        // On malformed inputs we will report alert and continue without updating the state
        Utils.logErrorAndNotifyUser(e);
        return state;
      }
    }
    default:
      return state;
  }
};
