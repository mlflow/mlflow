import { Empty, DangerIcon } from '@databricks/design-system';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

interface ArtifactViewErrorStateProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  description?: React.ReactNode;
  title?: React.ReactNode;
  error: unknown;
}

const getArtifactErrorDescription = (error: unknown) => {
  if (error instanceof ArtifactPreviewTooLargeError) return error.message;
  if (
    error instanceof ErrorWrapper &&
    ((error.getStatus() === 403 &&
      (error.textJson?.artifacts_presigned_only === true || error.textJson?.external_artifact_service === true)) ||
      (error.getStatus() === 400 && error.textJson?.invalid_artifact_root === true))
  ) {
    return error.getMessageField();
  }
  return undefined;
};

export const ArtifactViewErrorState = ({ description, title, error, ...props }: ArtifactViewErrorStateProps) => (
  <div
    css={{
      flex: 1,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
    }}
    {...props}
  >
    <Empty
      image={<DangerIcon />}
      title={
        title ?? (
          <FormattedMessage
            defaultMessage="Loading artifact failed"
            description="Run page > artifact view > error state > default error message"
          />
        )
      }
      description={getArtifactErrorDescription(error) ?? description}
    />
  </div>
);
