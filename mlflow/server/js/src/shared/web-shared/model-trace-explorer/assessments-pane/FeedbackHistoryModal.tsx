import { useMemo } from 'react';

import { Modal } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { FeedbackHistoryItem } from './FeedbackHistoryItem';
import type { Assessment, FeedbackAssessment } from '../ModelTrace.types';

// helper function to traverse the linked list of overridden
// assessments. this function handles cycles by keeping track
// of the assessments we've seen. the backend should prevent
// such cases from existing, but we should be defensive.
const flattenOverrides = (assessment: Assessment) => {
  const seen = new Set<string>();
  const flattened = [];

  let currentAssessment: Assessment | undefined = assessment;
  while (currentAssessment && !seen.has(currentAssessment.assessment_id)) {
    seen.add(currentAssessment.assessment_id);
    flattened.push(currentAssessment);
    currentAssessment = currentAssessment.overriddenAssessment;
  }

  return flattened;
};

export const FeedbackHistoryModal = ({
  isModalVisible,
  setIsModalVisible,
  feedback,
}: {
  isModalVisible: boolean;
  setIsModalVisible: (isModalVisible: boolean) => void;
  feedback: FeedbackAssessment;
}) => {
  const assessmentHistory = useMemo(() => flattenOverrides(feedback), [feedback]);

  return (
    <Modal
      componentId="shared.model-trace-explorer.feedback-history-modal"
      visible={isModalVisible}
      footer={null}
      onCancel={() => {
        setIsModalVisible(false);
      }}
      title={
        <FormattedMessage
          defaultMessage="Edit history"
          description="Title of a modal that shows the edit history of an assessment"
        />
      }
    >
      {assessmentHistory.map((assessment) =>
        'feedback' in assessment ? <FeedbackHistoryItem key={assessment.assessment_id} feedback={assessment} /> : null,
      )}
    </Modal>
  );
};
