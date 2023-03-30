import { CheckCircleIcon, ClockIcon, XCircleIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import { RunRowDateAndNestInfo } from '../../../utils/experimentPage.row-types';

const ErrorIcon = () => (
  <XCircleIcon css={(theme) => ({ color: theme.colors.textValidationDanger })} />
);

const FinishedIcon = () => (
  <CheckCircleIcon css={(theme) => ({ color: theme.colors.textValidationSuccess })} />
);

const getRunStatusIcon = (status: string) => {
  switch (status) {
    case 'FAILED':
    case 'KILLED':
      return <ErrorIcon />;
    case 'FINISHED':
      return <FinishedIcon />;
    case 'SCHEDULED':
      return <ClockIcon />; // This one is the same color as the link
    default:
      return null;
  }
};

export interface DateCellRendererProps {
  value: RunRowDateAndNestInfo;
}

export const DateCellRenderer = React.memo(
  ({ value: { startTime, referenceTime, runStatus } }: DateCellRendererProps) => {
    return (
      <span css={styles.cellWrapper} title={Utils.formatTimestamp(startTime)}>
        {getRunStatusIcon(runStatus)}
        {Utils.timeSinceStr(startTime, referenceTime)}
      </span>
    );
  },
);

const styles = {
  cellWrapper: (theme: Theme) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
    svg: {
      width: 14,
      height: 14,
    },
  }),
};
