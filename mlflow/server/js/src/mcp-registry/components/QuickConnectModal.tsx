import { useEffect, useMemo } from 'react';
import { Modal, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ConnectionSource } from '../types';
import type { MCPServer } from '../types';
import { resolveDisplayName, STATUS_TAG_COLOR } from '../utils';
import { deriveClientName } from '../installInstructions';
import { ConnectionInstructions } from './ConnectionInstructions';

export const QuickConnectModal = ({
  visible,
  server,
  onClose,
}: {
  visible: boolean;
  server: MCPServer;
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const endpoint = server.access_endpoints?.[0];
  const displayName = resolveDisplayName(server);
  const derivedName = useMemo(() => deriveClientName(server.name), [server.name]);

  useEffect(() => {
    if (visible && !endpoint) {
      onClose();
    }
  }, [visible, endpoint, onClose]);

  if (!endpoint) return null;

  return (
    <Modal
      componentId="mlflow.mcp_registry.quick_connect_modal"
      title={
        <FormattedMessage
          defaultMessage="Connect to {name}"
          description="Quick connect modal title"
          values={{ name: displayName }}
        />
      }
      visible={visible}
      onCancel={onClose}
      footer={null}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginTop: -theme.spacing.sm }}>
        <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
          <Typography.Text bold size="sm">
            <FormattedMessage
              defaultMessage="Access endpoint"
              description="Quick connect modal access endpoint label"
            />
          </Typography.Text>
          <Typography.Text color="secondary" size="sm">
            ·
          </Typography.Text>
          <Typography.Text size="sm">
            <Typography.Text bold size="sm">
              <FormattedMessage
                defaultMessage="Latest version:"
                description="Quick connect modal latest version label"
              />
            </Typography.Text>{' '}
            {server.latest_version ?? '—'}
          </Typography.Text>
          {endpoint.resolved_version?.status && (
            <Tag
              componentId="mlflow.mcp_registry.quick_connect.version_status"
              color={STATUS_TAG_COLOR[endpoint.resolved_version.status]}
            >
              {endpoint.resolved_version.status}
            </Tag>
          )}
        </span>
        <ConnectionInstructions source={ConnectionSource.ENDPOINT} endpoint={endpoint} derivedName={derivedName} />
      </div>
    </Modal>
  );
};
