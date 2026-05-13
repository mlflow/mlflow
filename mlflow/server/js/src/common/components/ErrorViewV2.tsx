import { Empty, WarningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * A simple wrapper over <Empty> component displaying an error message.
 */
export const ErrorViewV2 = ({
  error,
  image,
  title,
  button,
  className,
}: {
  error: Error;
  image?: React.ReactElement;
  title?: React.ReactElement;
  button?: React.ReactElement;
  className?: string;
}) => {
  return (
    <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }} className={className}>
      <Empty
        description={error.message}
        image={image ?? <WarningIcon />}
        title={
          title ?? (
            <FormattedMessage
              defaultMessage="Error"
              description="A generic error message for error state in MLflow UI"
            />
          )
        }
        button={button}
      />
    </div>
  );
};
