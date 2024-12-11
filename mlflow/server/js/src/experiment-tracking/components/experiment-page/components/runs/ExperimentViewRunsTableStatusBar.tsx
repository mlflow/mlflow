import { Spinner, Typography } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { FormattedMessage } from 'react-intl';

interface ExperimentViewRunsTableStatusBarProps {
  isLoading: boolean;
  allRunsCount: number;
}

// Strongifies the i18n string, used in <FormattedMessage> below
const strong = (text: string) => <strong>{text}</strong>;

export const ExperimentViewRunsTableStatusBar = ({
  isLoading,
  allRunsCount,
}: ExperimentViewRunsTableStatusBarProps) => (
  <div css={styles.statusBar}>
    <Typography.Text size="sm" color={isLoading ? 'secondary' : undefined}>
      <FormattedMessage
        // eslint-disable-next-line max-len
        defaultMessage="<strong>{length}</strong> matching {length, plural, =0 {runs} =1 {run} other {runs}}"
        // eslint-disable-next-line max-len
        description="Message for displaying how many runs match search criteria on experiment page"
        values={{
          strong,
          length: allRunsCount,
        }}
      />
    </Typography.Text>
    {isLoading && <Spinner size="small" />}
  </div>
);

const styles = {
  statusBar: (theme: Theme) => ({
    height: 28,
    display: 'flex',
    gap: 8,
    marginTop: -1,
    position: 'relative' as const,
    alignItems: 'center',
    borderTop: `1px solid ${theme.colors.border}`,
  }),
};
