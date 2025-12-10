import { Empty, KeyIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

const ApiKeysPage = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ padding: theme.spacing.md, flex: 1, overflow: 'auto' }}>
      <Empty
        image={<KeyIcon />}
        description={
          <FormattedMessage
            defaultMessage="API Keys management coming soon"
            description="Placeholder message for API keys page"
          />
        }
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
