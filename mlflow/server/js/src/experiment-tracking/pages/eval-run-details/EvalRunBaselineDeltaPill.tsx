import type { ReactNode } from 'react';
import { HoverCard, NewWindowIcon, Tag, Typography } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';

/** Compact pill with one scorer's saved run metric relative to the baseline run's saved metric. */
export const EvalRunBaselineDeltaPill = ({
  isRate,
  currentScore,
  baselineScore,
  baselineRunHref,
}: {
  isRate: boolean;
  currentScore?: number;
  baselineScore?: number;
  baselineRunHref: string;
}) => {
  const intl = useIntl();
  if (currentScore === undefined || !Number.isFinite(currentScore)) {
    return null;
  }
  // Rates are shown in percentage points.
  const delta =
    baselineScore !== undefined && Number.isFinite(baselineScore)
      ? (currentScore - baselineScore) * (isRate ? 100 : 1)
      : undefined;
  const label =
    delta === undefined
      ? '-'
      : delta === 0
        ? '+0.0'
        : intl.formatNumber(delta, { maximumFractionDigits: isRate ? 1 : 2, signDisplay: 'exceptZero' });
  const direction = delta === undefined || delta === 0 ? 'neutral' : delta > 0 ? 'improvement' : 'regression';
  const openBaselineLabel = intl.formatMessage({
    defaultMessage: 'Open evaluation baseline run in a new tab',
    description: 'Accessible label for opening the baseline evaluation run from a scorer tooltip',
  });
  const baselineLink = (chunks: ReactNode[]) => (
    <Link
      componentId="mlflow.eval-run-traces-v4.baseline-delta.open-baseline"
      to={baselineRunHref}
      target="_blank"
      rel="noreferrer"
      aria-label={openBaselineLabel}
      css={{ display: 'inline-flex', alignItems: 'center', gap: 2, textDecoration: 'underline' }}
    >
      {chunks}
      <NewWindowIcon aria-hidden css={{ fontSize: 14 }} />
    </Link>
  );

  return (
    <HoverCard
      trigger={
        <Tag
          componentId="mlflow.eval-run-traces-v4.baseline-delta"
          data-testid="assessment-baseline-delta-pill"
          data-baseline-delta-direction={direction}
          color={direction === 'improvement' ? 'lime' : direction === 'regression' ? 'coral' : undefined}
        >
          {label}
        </Tag>
      }
      content={
        <div css={{ padding: 4 }}>
          <Typography.Text>
            {delta === undefined ? (
              <FormattedMessage
                defaultMessage="<baselineLink>Evaluation baseline run</baselineLink> does not have this scorer."
                description="Tooltip shown when the baseline evaluation run has no saved metric for a scorer"
                values={{ baselineLink }}
              />
            ) : (
              <FormattedMessage
                defaultMessage="Compared to <baselineLink>evaluation baseline run</baselineLink>."
                description="Tooltip explaining that a scorer delta is relative to the baseline evaluation run"
                values={{ baselineLink }}
              />
            )}
          </Typography.Text>
        </div>
      }
      minWidth={280}
      maxWidth={420}
      align="start"
    />
  );
};
