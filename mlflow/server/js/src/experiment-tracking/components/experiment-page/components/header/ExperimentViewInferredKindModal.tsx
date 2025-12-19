import { Modal, Radio, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ExperimentKind } from '../../../../constants';
import { ExperimentKindDropdownLabels } from '../../../../utils/ExperimentKindUtils';

export const ExperimentViewInferredKindModal = ({
  onDismiss,
  onConfirm,
}: {
  onDismiss?: () => void;
  onConfirm: (kind: ExperimentKind) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [selectedKind, setSelectedKind] = useState<ExperimentKind>(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT);

  return (
    <Modal
      visible
      componentId="mlflow.experiment_view.header.experiment_kind_inference_modal"
      onCancel={onDismiss}
      title={
        <FormattedMessage
          defaultMessage="Choose experiment type"
          description="A title for the modal displayed when the experiment type could not be inferred"
        />
      }
      cancelText={
        <FormattedMessage
          defaultMessage="I'll choose later"
          description="A label for the dismissal button in the modal displayed when the experiment type could not be inferred"
        />
      }
      okText={
        <FormattedMessage
          defaultMessage="Confirm"
          description="A label for the confirmation button in the modal displayed when the experiment type could not be inferred"
        />
      }
      onOk={() => onConfirm(selectedKind)}
    >
      <Typography.Paragraph>
        <FormattedMessage
          defaultMessage="We support multiple experiment types, each with its own set of features. Please select the type you'd like to use. You can change this later if needed."
          description="Popover message displayed when the experiment type could not not inferred"
        />
      </Typography.Paragraph>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
        <Radio
          checked={selectedKind === ExperimentKind.GENAI_DEVELOPMENT}
          onChange={() => setSelectedKind(ExperimentKind.GENAI_DEVELOPMENT)}
        >
          <FormattedMessage {...ExperimentKindDropdownLabels[ExperimentKind.GENAI_DEVELOPMENT]} />
        </Radio>
        <Radio
          checked={selectedKind === ExperimentKind.CUSTOM_MODEL_DEVELOPMENT}
          onChange={() => setSelectedKind(ExperimentKind.CUSTOM_MODEL_DEVELOPMENT)}
        >
          <FormattedMessage {...ExperimentKindDropdownLabels[ExperimentKind.CUSTOM_MODEL_DEVELOPMENT]} />
        </Radio>
      </div>
    </Modal>
  );
};
