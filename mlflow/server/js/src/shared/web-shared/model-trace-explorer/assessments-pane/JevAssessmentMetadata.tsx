import { Fragment } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import type { FeedbackAssessment } from '../ModelTrace.types';

const parseMap = (value: string | undefined): Record<string, unknown> => {
  if (!value) return {};
  try {
    const result: unknown = JSON.parse(value);
    return result && typeof result === 'object' && !Array.isArray(result) ? (result as Record<string, unknown>) : {};
  } catch {
    return {};
  }
};

export const JevAssessmentMetadata = ({ metadata }: { metadata: FeedbackAssessment['metadata'] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (!metadata?.['jev.model']) return null;

  const probabilities = Object.entries(parseMap(metadata['jev.probabilities'])).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === 'number' && Number.isFinite(entry[1]) && entry[1] >= 0 && entry[1] <= 1,
  );
  const legend = parseMap(metadata['jev.legend']);
  const formatProbability = (value: string | undefined) => {
    if (value === undefined || value.trim() === '') return undefined;
    const numericValue = Number(value);
    return Number.isFinite(numericValue) && numericValue >= 0 && numericValue <= 1
      ? intl.formatNumber(numericValue, { style: 'percent', maximumFractionDigits: 2 })
      : undefined;
  };
  const probability = formatProbability(metadata['jev.probability']);
  const confidence = formatProbability(metadata['jev.confidence']);

  if (!probability && !confidence && probabilities.length === 0) return null;

  return (
    <dl
      css={{
        margin: 0,
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr) auto',
        gap: theme.spacing.xs,
        '& dt': { color: theme.colors.textSecondary },
        '& dd': { margin: 0, textAlign: 'right' },
      }}
    >
      {probability && (
        <>
          <dt>
            <FormattedMessage defaultMessage="Probability of true" description="Jev noul probability label" />
          </dt>
          <dd>{probability}</dd>
        </>
      )}
      {confidence && (
        <>
          <dt>
            <FormattedMessage defaultMessage="Confidence" description="Jev confidence label" />
          </dt>
          <dd>{confidence}</dd>
        </>
      )}
      {probabilities.map(([label, value]) => (
        <Fragment key={label}>
          <dt>{typeof legend[label] === 'string' ? `${label}: ${legend[label]}` : label}</dt>
          <dd>{intl.formatNumber(value, { style: 'percent', maximumFractionDigits: 2 })}</dd>
        </Fragment>
      ))}
    </dl>
  );
};
