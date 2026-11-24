import { ChecklistIcon, Empty } from '@databricks/design-system';

/**
 * Centered onboarding empty state shared by the review queue panels (no queues,
 * no queue selected, empty queue). Wraps the design-system `Empty` in the
 * centered container documented in the frontend CLAUDE.md so the three call
 * sites stay consistent.
 */
export const ReviewQueueEmptyState = ({
  title,
  description,
  button,
}: {
  title: React.ReactNode;
  description: React.ReactNode;
  button?: React.ReactNode;
}) => (
  <div
    css={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      minHeight: 400,
      width: '100%',
      '& > div': {
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      },
    }}
  >
    <Empty image={<ChecklistIcon />} title={title} description={description} button={button} />
  </div>
);
