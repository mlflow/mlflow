import { Button, LoadingIcon, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';

export interface CancelIssueDetectionButtonProps {
  componentId: string;
  /** Ends the detection in progress. Unset while a launch has yet to create a run, which disables the button. */
  onCancel?: () => Promise<void>;
  size?: 'small' | 'middle';
  /** The page already says when to cancel, so the dialog only says what cancelling does. */
  isWhenToCancelShownOnPage?: boolean;
}

/**
 * Stands in for "Detect issues" while a detection runs, so a run whose job stopped without
 * reporting it doesn't leave the button disabled with no way to start another. It confirms
 * first: a healthy run takes a while, and cancelling one throws its work away.
 */
export const CancelIssueDetectionButton = ({
  componentId,
  onCancel,
  size,
  isWhenToCancelShownOnPage = false,
}: CancelIssueDetectionButtonProps) => {
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
        icon={isCancelling ? <LoadingIcon /> : undefined}
        onClick={() => setIsConfirmOpen(true)}
      >
        <FormattedMessage
          defaultMessage="Cancel detection"
          description="Experiment issues tab > Action to cancel the issue detection in progress"
        />
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
        <Typography.Text css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Cancelling discards the analysis done so far. Any issues already recorded are kept."
            description="Experiment issues tab > Explanation of what cancelling issue detection discards"
          />
        </Typography.Text>
        {!isWhenToCancelShownOnPage && (
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="A detection run takes a while. Cancel only if it seems to have stopped making progress."
              description="Experiment issues tab > Guidance on when cancelling issue detection is appropriate"
            />
          </Typography.Text>
        )}
      </Modal>
    </>
  );
};
