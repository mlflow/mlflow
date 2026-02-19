import { useState } from 'react';

import { Alert, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode, type AssessmentError } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';

export const FeedbackErrorItem = ({ error }: { error: AssessmentError }) => {
  const { theme } = useDesignSystemTheme();
  const [isModalVisible, setIsModalVisible] = useState(false);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Alert
        type="error"
        closable={false}
        message={error.error_code}
        componentId="shared.model-trace-explorer.feedback-error-item"
        description={
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <span>{error.error_message}</span>
            {error.stack_trace && (
              <Typography.Link
                componentId="shared.model-trace-explorer.feedback-error-item-stack-trace-link"
                onClick={() => setIsModalVisible(true)}
              >
                <FormattedMessage
                  defaultMessage="View stack trace"
                  description="Link to view the stack trace for an assessment error"
                />
              </Typography.Link>
            )}
          </div>
        }
      />
      {error.stack_trace && (
        <Modal
          title={
            <FormattedMessage
              defaultMessage="Error stack trace"
              description="Title of the assessment error stack trace modal"
            />
          }
          visible={isModalVisible}
          componentId="shared.model-trace-explorer.feedback-error-stack-trace-modal"
          footer={null}
          onCancel={() => setIsModalVisible(false)}
        >
          <ModelTraceExplorerCodeSnippet
            data={JSON.stringify(error.stack_trace)}
            title=""
            initialRenderMode={CodeSnippetRenderMode.TEXT}
          />
        </Modal>
      )}
    </div>
  );
};
