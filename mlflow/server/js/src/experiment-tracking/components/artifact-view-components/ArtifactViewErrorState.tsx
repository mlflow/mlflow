import { Empty, DangerIcon } from '@databricks/design-system';
import React from 'react';
import { FormattedMessage } from 'react-intl';

interface ArtifactViewErrorStateProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  description?: React.ReactNode;
  title?: React.ReactNode;
}

export const ArtifactViewErrorState = ({ description, title, ...props }: ArtifactViewErrorStateProps) => (
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
      description={description}
    />
  </div>
);
