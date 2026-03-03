import { CopyIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { CodeSnippet } from '@databricks/web-shared/snippet';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';

function getCodeSnippet(datasetName: string): string {
  return `from mlflow.genai.datasets import get_dataset

dataset = get_dataset(name="${datasetName}")
dataset.merge_records([
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"expected_response": "An ML platform"},
    }
])`;
}

export const AddViaCodeModal = ({
  visible,
  onCancel,
  datasetName,
}: {
  visible: boolean;
  onCancel: () => void;
  datasetName: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const code = getCodeSnippet(datasetName);

  return (
    <Modal
      componentId="mlflow.add-via-code-modal"
      visible={visible}
      onCancel={onCancel}
      title={
        <FormattedMessage
          defaultMessage="Add records via code"
          description="Add via code modal title for dataset records"
        />
      }
      footer={null}
      zIndex={theme.options.zIndexBase + 20}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Use the MLflow Python SDK to add records to your dataset programmatically."
            description="Description text in the add via code modal"
          />
        </Typography.Text>

        <div css={{ position: 'relative' }}>
          <CopyButton
            componentId="mlflow.add-via-code-modal.copy"
            css={{
              zIndex: 1,
              position: 'absolute',
              top: theme.spacing.xs,
              right: theme.spacing.xs,
            }}
            showLabel={false}
            copyText={code}
            icon={<CopyIcon />}
          />
          <CodeSnippet showLineNumbers language="python" theme={theme.isDarkMode ? 'duotoneDark' : 'light'}>
            {code}
          </CodeSnippet>
        </div>

        <Typography.Text color="secondary" size="sm">
          <FormattedMessage
            defaultMessage="For more information, visit the {docsLink}."
            description="Documentation link text in the add via code modal"
            values={{
              docsLink: (
                <Typography.Link
                  componentId="mlflow.add-via-code-modal.docs-link"
                  href="https://mlflow.org/docs/latest/genai/datasets/sdk-guide/#adding-records-to-a-dataset"
                  openInNewTab
                >
                  <FormattedMessage
                    defaultMessage="MLflow Datasets documentation"
                    description="Link text for datasets documentation"
                  />
                </Typography.Link>
              ),
            }}
          />
        </Typography.Text>
      </div>
    </Modal>
  );
};
