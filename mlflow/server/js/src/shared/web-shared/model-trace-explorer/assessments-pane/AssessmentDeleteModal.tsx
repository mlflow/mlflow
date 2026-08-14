import { useCallback } from 'react';

import { Modal } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { Assessment } from '../ModelTrace.types';
import { useDeleteAssessment } from '../hooks/useDeleteAssessment';

export const AssessmentDeleteModal = ({
  assessment,
  isModalVisible,
  setIsModalVisible,
}: {
  assessment: Assessment;
  isModalVisible: boolean;
  setIsModalVisible: (isModalVisible: boolean) => void;
}) => {
  const { deleteAssessmentMutation, isLoading } = useDeleteAssessment({
    assessment,
    onSettled: () => {
      setIsModalVisible(false);
    },
  });

  const handleDelete = useCallback(() => {
    deleteAssessmentMutation();
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
