import { CheckCircleIcon, Tag, useDesignSystemTheme, XCircleIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import {
  getPairwiseScoreCounts,
  PAIRWISE_SCORE_LOSE_VALUE,
  PAIRWISE_SCORE_WIN_VALUE,
  type PairwiseScoreValue,
} from '../utils/pairwiseReview';

export interface PairwiseScoreDisplayValueProps {
  value: PairwiseScoreValue;
  count?: number;
}

export const PairwiseScoreDisplayValue = ({ value, count = 1 }: PairwiseScoreDisplayValueProps) => {
  const { theme } = useDesignSystemTheme();
  const isWin = value === PAIRWISE_SCORE_WIN_VALUE;
  const Icon = isWin ? CheckCircleIcon : XCircleIcon;

  return (
    <Tag
      componentId="mlflow.traces-v4.pairwise-score.value"
      color={isWin ? 'lime' : 'coral'}
      css={{ display: 'inline-flex', maxWidth: '100%', minWidth: theme.spacing.md, marginRight: 0 }}
    >
      <span
        css={{
          alignItems: 'center',
          color: isWin
            ? theme.isDarkMode
              ? theme.colors.green400
              : theme.colors.green700
            : theme.isDarkMode
              ? theme.colors.red400
              : theme.colors.red700,
          display: 'inline-flex',
          fontSize: theme.typography.fontSizeSm,
          lineHeight: theme.typography.lineHeightBase,
          overflow: 'hidden',
          paddingInline: theme.spacing.xs,
          textOverflow: 'ellipsis',
          textWrap: 'nowrap',
        }}
      >
        <Icon
          css={{
            fontSize: theme.typography.fontSizeSm,
            marginRight: theme.spacing.sm,
            svg: { width: theme.typography.fontSizeSm, height: theme.typography.fontSizeSm },
          }}
        />
        {isWin ? (
          <FormattedMessage defaultMessage="Win" description="Label for a winning pairwise preference assessment" />
        ) : (
          <FormattedMessage defaultMessage="Lose" description="Label for a losing pairwise preference assessment" />
        )}
        {count > 1 ? ` (${count})` : null}
      </span>
    </Tag>
  );
};

export interface PairwiseScoreCellProps {
  trace: ModelTraceInfoV3;
}

export const PairwiseScoreCell = ({ trace }: PairwiseScoreCellProps) => {
  const { theme } = useDesignSystemTheme();
  const { wins, losses } = getPairwiseScoreCounts(trace);

  if (!wins && !losses) {
    return null;
  }

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      {wins > 0 && <PairwiseScoreDisplayValue value={PAIRWISE_SCORE_WIN_VALUE} count={wins} />}
      {losses > 0 && <PairwiseScoreDisplayValue value={PAIRWISE_SCORE_LOSE_VALUE} count={losses} />}
    </div>
  );
};
