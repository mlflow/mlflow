import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { keys, pickBy } from 'lodash';
import React, { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { MlflowService } from '../../sdk/MlflowService';
import { Modal, Typography } from '@databricks/design-system';

export const TracesViewDeleteTraceModal = ({
  experimentIds,
  visible,
  rowSelection,
  setRowSelection,
  handleClose,
  refreshTraces,
}: {
  experimentIds: string[];
  visible: boolean;
  rowSelection: { [id: string]: boolean };
  setRowSelection: (rowSelection: { [id: string]: boolean }) => void;
  handleClose: () => void;
  refreshTraces: () => void;
}) => {
  const intl = useIntl();
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const tracesToDelete = keys(pickBy(rowSelection, (value) => value));

  const submitDeleteTraces = async () => {
    try {
      // TODO: Add support for deleting traces from multiple experiments
      // The trace data contains the experiment ID, so we simply need to
      // pass the trace data instead of just the trace IDs.
      await MlflowService.deleteTraces(experimentIds[0] ?? '', tracesToDelete);

      // reset row selection and refresh traces
      setRowSelection({});
      refreshTraces();
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
      componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewdeletetracemodal.tsx_62"
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
