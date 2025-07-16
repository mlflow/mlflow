import { useCallback } from 'react';

import { Modal } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useMutation } from '@databricks/web-shared/query-client';

import type { Assessment } from '../ModelTrace.types';
import { displayErrorNotification } from '../ModelTraceExplorer.utils';
import { deleteAssessment } from '../api';
import { useModelTraceInfoRefetchContext } from '../trace-context/ModelTraceInfoRefetchContext';

export const AssessmentDeleteModal = ({
  assessment,
  isModalVisible,
  setIsModalVisible,
}: {
  assessment: Assessment;
  isModalVisible: boolean;
  setIsModalVisible: (isModalVisible: boolean) => void;
}) => {
  const intl = useIntl();
  const { refetchTraceInfo } = useModelTraceInfoRefetchContext();

  const { mutate: deleteAssessmentMutation, isLoading } = useMutation({
    mutationFn: () => deleteAssessment({ traceId: assessment.trace_id, assessmentId: assessment.assessment_id }),
    onSuccess: () => {
      refetchTraceInfo?.();
    },
    onError: (error) => {
      displayErrorNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Failed to delete assessment. Error: {error}',
            description: 'Error message when deleting an assessment fails.',
          },
          {
            error: error instanceof Error ? error.message : String(error),
          },
        ),
      );
    },
    onSettled: () => {
      setIsModalVisible(false);
    },
  });

  const handleDelete = useCallback(async () => {
    await deleteAssessmentMutation();
  }, [deleteAssessmentMutation]);

  return (
    <Modal
      componentId="shared.model-trace-explorer.assessment-delete-modal"
      visible={isModalVisible}
      onOk={handleDelete}
      okButtonProps={{ danger: true, loading: isLoading }}
      okText={<FormattedMessage defaultMessage="Delete" description="Delete assessment modal button text" />}
      onCancel={() => {
        setIsModalVisible(false);
      }}
      cancelText={<FormattedMessage defaultMessage="Cancel" description="Delete assessment modal cancel button text" />}
      confirmLoading={isLoading}
      title={<FormattedMessage defaultMessage="Delete assessment" description="Delete assessments modal title" />}
    >
      <FormattedMessage
        defaultMessage="Are you sure you want to delete this assessment?"
        description="Delete assessments modal confirmation text"
      />
    </Modal>
  );
};
