import { useState } from 'react';

import { Button, Checkbox, Modal, Typography, UserGroupIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

const CID = 'shared.model-trace-explorer.review-queue.assign-button';

// POC ONLY: hardcoded reviewers to show what the assign UI will expose.
// The button is intentionally non-functional (presentational) — it
// demonstrates the developer-side "assign this trace to reviewers"
// affordance for the Review Queue feedback prototype.
const POC_REVIEWERS = [
  { id: 'sme1@example.com', label: 'Priya (sme1@example.com)' },
  { id: 'sme2@example.com', label: 'Marco (sme2@example.com)' },
  { id: 'sme3@example.com', label: 'Dana (sme3@example.com)' },
];

export const ReviewQueueAssignButton = () => {
  const { theme } = useDesignSystemTheme();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);

  const toggle = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  return (
    <>
      <Button componentId={`${CID}.open`} size="small" icon={<UserGroupIcon />} onClick={() => setOpen(true)}>
        <FormattedMessage
          defaultMessage="Assign to reviewer"
          description="Button on the trace detail header to assign the trace to reviewers"
        />
      </Button>
      {open && (
        <Modal
          visible
          componentId={CID}
          title={
            <FormattedMessage
              defaultMessage="Assign to reviewer"
              description="Title of the assign-to-reviewer modal on the trace detail page"
            />
          }
          onCancel={() => setOpen(false)}
          footer={
            <>
              <Button componentId={`${CID}.cancel`} onClick={() => setOpen(false)}>
                <FormattedMessage defaultMessage="Cancel" description="Cancel button in the assign-to-reviewer modal" />
              </Button>
              {/* POC: intentionally a no-op; just closes the modal. */}
              <Button
                componentId={`${CID}.assign`}
                type="primary"
                disabled={selected.length === 0}
                onClick={() => setOpen(false)}
              >
                <FormattedMessage
                  defaultMessage="Assign ({count})"
                  description="Confirm button in the assign-to-reviewer modal, with reviewer count"
                  values={{ count: selected.length }}
                />
              </Button>
            </>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Selected reviewers will get this trace in their Review Queue."
                description="Explanation in the assign-to-reviewer modal"
              />
            </Typography.Text>
            {POC_REVIEWERS.map((r) => (
              <Checkbox
                key={r.id}
                componentId={`${CID}.reviewer`}
                isChecked={selected.includes(r.id)}
                onChange={() => toggle(r.id)}
              >
                {r.label}
              </Checkbox>
            ))}
          </div>
        </Modal>
      )}
    </>
  );
};
