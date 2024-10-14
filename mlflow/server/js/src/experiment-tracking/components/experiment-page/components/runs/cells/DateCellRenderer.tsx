import { Theme } from '@emotion/react';
import React from 'react';
import Utils from '../../../../../../common/utils/Utils';
import { RunRowDateAndNestInfo } from '../../../utils/experimentPage.row-types';
import { RunStatusIcon } from '../../../../RunStatusIcon';

export interface DateCellRendererProps {
  value: RunRowDateAndNestInfo;
}

export const DateCellRenderer = React.memo(({ value }: DateCellRendererProps) => {
  const { startTime, referenceTime, runStatus } = value || {};
  if (!startTime) {
    return <>-</>;
  }
  return (
    <span css={styles.cellWrapper} title={Utils.formatTimestamp(startTime)}>
      <RunStatusIcon status={runStatus} />
      {Utils.timeSinceStr(startTime, referenceTime)}
    </span>
  );
});

const styles = {
  cellWrapper: (theme: Theme) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing.sm,
  }),
};
