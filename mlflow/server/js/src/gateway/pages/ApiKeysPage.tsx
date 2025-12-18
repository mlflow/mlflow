import { useState } from 'react';
import { Button, PlusIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ApiKeysList, CreateApiKeyModal } from '../components/api-keys';

const ApiKeysPage = () => {
  const { theme } = useDesignSystemTheme();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: theme.spacing.md,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0 }}>
          <FormattedMessage defaultMessage="API Keys" description="API Keys page title" />
        </Typography.Title>
        <Button
          componentId="mlflow.gateway.api-keys.create-button"
          type="primary"
          icon={<PlusIcon />}
          onClick={() => setIsCreateModalOpen(true)}
        >
          <FormattedMessage
            defaultMessage="Create API key"
            description="Gateway > API keys page > Create API key button"
          />
        </Button>
      </div>

      <div css={{ flex: 1, overflow: 'auto', padding: theme.spacing.md }}>
        <ApiKeysList />
      </div>

      <CreateApiKeyModal
        open={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSuccess={() => setIsCreateModalOpen(false)}
      />
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, ApiKeysPage);
