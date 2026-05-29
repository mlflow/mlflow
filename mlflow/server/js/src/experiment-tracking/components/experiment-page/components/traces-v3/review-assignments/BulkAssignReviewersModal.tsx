import { useMemo, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

import {
  Alert,
  Button,
  FormUI,
  Input,
  Modal,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { useCurrentUserQuery } from '../../../../../../account/hooks';
import { useBulkAssignReviewers } from './useBulkAssignReviewers';

const COMPONENT_ID = 'mlflow.traces.review-assignments.assign-modal';

/**
 * Splits the free-form reviewer textarea (one reviewer per line or
 * comma-separated) into a de-duplicated, trimmed list.
 */
export const parseReviewers = (raw: string): string[] => {
  const seen = new Set<string>();
  for (const token of raw.split(/[\n,]/)) {
    const reviewer = token.trim();
    if (reviewer) {
      seen.add(reviewer);
    }
  }
  return Array.from(seen);
};

export const BulkAssignReviewersModal = ({
  experimentId,
  traceIds,
  onClose,
}: {
  experimentId: string;
  traceIds: string[];
  onClose: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [reviewersText, setReviewersText] = useState('');
  const { data: currentUserData } = useCurrentUserQuery();
  const assigner = currentUserData?.user?.username ?? '';

  const reviewers = useMemo(() => parseReviewers(reviewersText), [reviewersText]);
  const [failedCount, setFailedCount] = useState(0);
  const { mutate: bulkAssign, isLoading, error, reset } = useBulkAssignReviewers();

  const handleClose = () => {
    reset();
    setReviewersText('');
    setFailedCount(0);
    onClose();
  };

  const handleAssign = () => {
    // Clear both stale outcome banners (hard error + partial-failure) on retry.
    reset();
    setFailedCount(0);
    bulkAssign(
      { experimentId, traceIds, reviewers, assigner },
      {
        onSuccess: (result) => {
          // Per-row failures come back in the response body, not as an
          // error. Keep the modal open and surface the count when any
          // pair failed; close cleanly otherwise. The created / existing
          // buckets aren't surfaced in v1 (re-assigning is idempotent, so
          // an already-assigned pair is a no-op, not an error); a richer
          // success summary is deferred to the Reviews-page stack.
          if (result.failed.length > 0) {
            setFailedCount(result.failed.length);
          } else {
            handleClose();
          }
        },
      },
    );
  };

  const canAssign = reviewers.length > 0 && traceIds.length > 0 && Boolean(assigner) && !isLoading;
  const assignTooltip = !assigner
    ? intl.formatMessage({
        defaultMessage: 'Sign in to assign reviewers.',
        description: 'Review assignments: tooltip when no current user is available to record as the assigner',
      })
    : undefined;

  return (
    <Modal
      visible
      componentId={COMPONENT_ID}
      title={
        <FormattedMessage
          defaultMessage="Assign reviewers"
          description="Review assignments: title of the bulk assign reviewers modal"
        />
      }
      onCancel={handleClose}
      footer={
        <>
          <Button componentId={`${COMPONENT_ID}.cancel`} onClick={handleClose}>
            <FormattedMessage
              defaultMessage="Cancel"
              description="Review assignments: cancel button in the assign reviewers modal"
            />
          </Button>
          <Tooltip componentId={`${COMPONENT_ID}.assign-tooltip`} content={assignTooltip}>
            {/* A disabled button doesn't emit pointer events, so wrap it
                in a span for the tooltip to fire on hover. */}
            <span>
              <Button
                componentId={`${COMPONENT_ID}.assign`}
                type="primary"
                onClick={handleAssign}
                disabled={!canAssign}
                loading={isLoading}
              >
                <FormattedMessage
                  defaultMessage="Assign ({count})"
                  description="Review assignments: confirm button in the assign reviewers modal, with reviewer count"
                  values={{ count: reviewers.length }}
                />
              </Button>
            </span>
          </Tooltip>
        </>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Assigning {traceCount, plural, one {# trace} other {# traces}} for review."
            description="Review assignments: summary of how many traces the assignment applies to"
            values={{ traceCount: traceIds.length }}
          />
        </Typography.Text>
        <FormUI.Label htmlFor={`${COMPONENT_ID}.reviewers-input`}>
          <FormattedMessage
            defaultMessage="Reviewers"
            description="Review assignments: label for the reviewers input"
          />
        </FormUI.Label>
        <FormUI.Hint>
          <FormattedMessage
            defaultMessage="One reviewer per line, or comma-separated (typically email)."
            description="Review assignments: hint describing how to enter reviewers"
          />
        </FormUI.Hint>
        <Input.TextArea
          componentId={`${COMPONENT_ID}.reviewers-input`}
          id={`${COMPONENT_ID}.reviewers-input`}
          value={reviewersText}
          onChange={(e) => setReviewersText(e.target.value)}
          autoSize={{ minRows: 4, maxRows: 10 }}
          autoComplete="off"
        />
        {error && <Alert componentId={`${COMPONENT_ID}.error`} type="error" closable={false} message={error.message} />}
        {failedCount > 0 && (
          <Alert
            componentId={`${COMPONENT_ID}.partial-failure`}
            type="warning"
            closable={false}
            message={intl.formatMessage(
              {
                defaultMessage: '{failedCount, plural, one {# assignment} other {# assignments}} could not be created.',
                description: 'Review assignments: warning when some assignments in the batch failed to create',
              },
              { failedCount },
            )}
          />
        )}
      </div>
    </Modal>
  );
};
