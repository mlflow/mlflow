import { useDesignSystemTheme, Empty } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ErrorBoundary } from 'react-error-boundary';

import { useParams } from '../../../common/utils/RoutingUtils';
import { LabelSchemasContentContainer } from './LabelSchemasContentContainer';

const ErrorFallback = ({ error }: { error?: Error }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
      }}
    >
      <Empty
        title={
          <FormattedMessage
            defaultMessage="Unable to load label schemas"
            description="Error message when label schemas page fails to load"
          />
        }
        description={error ? <span>{error.message}</span> : null}
      />
    </div>
  );
};

const LabelSchemasPage = () => {
  const { experimentId } = useParams();
  if (!experimentId) {
    return null;
  }
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <LabelSchemasContentContainer experimentId={experimentId} />
    </ErrorBoundary>
  );
};

export default LabelSchemasPage;
