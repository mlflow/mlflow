import { Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { Dataset } from '../hooks/useDatasetsQueries';

export interface DatasetMetadataModalProps {
  dataset: Dataset;
  visible: boolean;
  onClose: () => void;
}

/**
 * Render `json` pretty-printed when it parses as JSON; otherwise the raw string. The
 * backend returns these as serialized JSON strings, but a malformed/legacy value should
 * still show *something* rather than disappear.
 */
const formatJson = (json: string): string => {
  try {
    return JSON.stringify(JSON.parse(json), null, 2);
  } catch {
    return json;
  }
};

export const DatasetMetadataModal = ({ dataset, visible, onClose }: DatasetMetadataModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const hasDigest = Boolean(dataset.digest);
  const hasSchema = Boolean(dataset.schema);
  const hasProfile = Boolean(dataset.profile);
  const hasAny = hasDigest || hasSchema || hasProfile;

  const preBlockCss = {
    backgroundColor: theme.colors.backgroundSecondary,
    padding: theme.spacing.sm,
    borderRadius: theme.borders.borderRadiusMd,
    fontFamily: 'monospace',
    fontSize: theme.typography.fontSizeSm,
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    maxHeight: 320,
    overflow: 'auto' as const,
    margin: 0,
  };

  return (
    <Modal
      componentId="mlflow.eval-datasets-v2.detail.metadata-modal"
      visible={visible}
      onCancel={onClose}
      onOk={onClose}
      okText={intl.formatMessage({
        defaultMessage: 'Done',
        description: 'Confirm-button text for the V2 evaluation dataset metadata modal',
      })}
      cancelText={null}
      title={
        <FormattedMessage
          defaultMessage="Dataset metadata"
          description="Title for the V2 evaluation dataset metadata modal"
        />
      }
      size="wide"
    >
      {!hasAny ? (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No additional metadata is available for this dataset."
            description="Body shown in the V2 evaluation dataset metadata modal when digest/schema/profile are all empty"
          />
        </Typography.Text>
      ) : (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          {hasDigest && (
            <div>
              <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Digest"
                  description="Label for the digest field in the V2 evaluation dataset metadata modal"
                />
              </Typography.Text>
              <Typography.Text code>{dataset.digest ?? ''}</Typography.Text>
            </div>
          )}
          {hasSchema && (
            <div>
              <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Schema"
                  description="Label for the schema field in the V2 evaluation dataset metadata modal"
                />
              </Typography.Text>
              <pre css={preBlockCss}>{formatJson(dataset.schema ?? '')}</pre>
            </div>
          )}
          {hasProfile && (
            <div>
              <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
                <FormattedMessage
                  defaultMessage="Profile"
                  description="Label for the profile field in the V2 evaluation dataset metadata modal"
                />
              </Typography.Text>
              <pre css={preBlockCss}>{formatJson(dataset.profile ?? '')}</pre>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
};
