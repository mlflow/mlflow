import React, { useState } from 'react';

import { Modal, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { RunEvaluationTracesDataEntry } from '../types';

export const GenAiDeleteTraceModal = ({
  experimentIds,
  visible,
  selectedTraces,
  handleClose,
  deleteTraces,
}: {
  experimentIds: string[];
  visible: boolean;
  selectedTraces: RunEvaluationTracesDataEntry[];
  handleClose: () => void;
  deleteTraces: (experimentId: string, traceIds: string[]) => Promise<void>;
}) => {
  const intl = useIntl();
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const tracesToDelete = selectedTraces.map((trace) => trace.evaluationId);

  const submitDeleteTraces = async () => {
    try {
      await deleteTraces(experimentIds[0] ?? '', tracesToDelete);
      handleClose();
    } catch (e: any) {
      setErrorMessage(
        intl.formatMessage({
          defaultMessage: 'An error occured while attempting to delete traces. Please refresh the page and try again.',
          description: 'Experiment page > traces view controls > Delete traces modal > Error message',
        }),
      );
    }
    setIsLoading(false);
  };

  const handleOk = () => {
    submitDeleteTraces();
    setIsLoading(true);
  };

  return (
    <Modal
      componentId="eval-tab.delete_traces-modal"
      title={
        <FormattedMessage
          defaultMessage="{count, plural, one {Delete Trace} other {Delete Traces}}"
          description="Experiment page > traces view controls > Delete traces modal > Title"
          values={{ count: tracesToDelete.length }}
        />
      }
      visible={visible}
      onCancel={handleClose}
      okText={
        <FormattedMessage
          defaultMessage="Delete {count, plural, one { # trace } other { # traces }}"
          description="Experiment page > traces view controls > Delete traces modal > Delete button"
          values={{ count: tracesToDelete.length }}
        />
      }
      onOk={handleOk}
      okButtonProps={{ loading: isLoading, danger: true }}
    >
      {errorMessage && <Typography.Paragraph color="error">{errorMessage}</Typography.Paragraph>}
      <Typography.Paragraph>
        <Typography.Text bold>
          <FormattedMessage
            defaultMessage="{count, plural, one { # trace } other { # traces }} will be deleted."
            description="Experiment page > traces view controls > Delete traces modal > Confirmation message title"
            values={{
              count: tracesToDelete.length,
            }}
          />
        </Typography.Text>
      </Typography.Paragraph>
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage="Deleted traces cannot be restored. Are you sure you want to proceed?"
          description="Experiment page > traces view controls > Delete traces modal > Confirmation message"
        />
      </Typography.Paragraph>
    </Modal>
  );
};
