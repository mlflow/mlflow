import { Button, Modal, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';

export interface CancelIssueDetectionButtonProps {
  componentId: string;
  /** Ends the detection in progress. Unset while a launch has yet to create a run, which disables the button. */
  onCancel?: () => Promise<void>;
  size?: 'small' | 'middle';
}

/**
 * Stands in for "Detect issues" while a detection runs, so a run whose job stopped without
 * reporting it doesn't leave the button disabled with no way to start another. It confirms
 * first: a healthy run takes a while, and cancelling one throws its work away.
 */
export const CancelIssueDetectionButton = ({ componentId, onCancel, size }: CancelIssueDetectionButtonProps) => {
  const { theme } = useDesignSystemTheme();
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);

  const confirmCancel = async () => {
    if (!onCancel) {
      return;
    }
    setIsCancelling(true);
    try {
      await onCancel();
    } finally {
      setIsCancelling(false);
      setIsConfirmOpen(false);
    }
  };

  return (
    <>
      <Button
        componentId={componentId}
        size={size}
        disabled={!onCancel || isCancelling}
        onClick={() => setIsConfirmOpen(true)}
      >
        <span css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
          {/*
            The spinner stands for the run still going, not for the cancel request, so it spins
            throughout and the button stays clickable. `LoadingIcon` is a static glyph, which
            read as a stalled spinner.
          */}
          <Spinner size="small" inheritColor />
          <FormattedMessage
            defaultMessage="Cancel detection"
            description="Experiment issues tab > Action to cancel the issue detection in progress"
          />
        </span>
      </Button>
      <Modal
        componentId={`${componentId}.confirm`}
        visible={isConfirmOpen}
        onCancel={() => setIsConfirmOpen(false)}
        title={
          <FormattedMessage
            defaultMessage="Cancel issue detection"
            description="Experiment issues tab > Title of the dialog confirming cancellation of issue detection"
          />
        }
        okText={
          <FormattedMessage
            defaultMessage="Cancel detection"
            description="Experiment issues tab > Dialog action confirming that issue detection should be cancelled"
          />
        }
        cancelText={
          <FormattedMessage
            defaultMessage="Keep running"
            description="Experiment issues tab > Dialog action leaving the issue detection running"
          />
        }
        okButtonProps={{ danger: true, loading: isCancelling }}
        onOk={confirmCancel}
      >
        <Typography.Text>
          <FormattedMessage
            defaultMessage="Cancelling discards the analysis done so far. Any issues already recorded are kept."
            description="Experiment issues tab > Explanation of what cancelling issue detection discards"
          />
        </Typography.Text>
      </Modal>
    </>
  );
};
