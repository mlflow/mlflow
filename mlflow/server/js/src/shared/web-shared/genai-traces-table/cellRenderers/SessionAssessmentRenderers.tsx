import React from 'react';

import type { ThemeType } from '@databricks/design-system';
import { HoverCard, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

interface PassFailBarProps {
  passCount: number;
  totalCount: number;
}

/**
 * Renders a bar chart showing pass/fail ratio with count label.
 */
export const PassFailBar: React.FC<PassFailBarProps> = ({ passCount, totalCount }) => {
  const { theme } = useDesignSystemTheme();

  const passBarColor = theme.isDarkMode ? theme.colors.green400 : theme.colors.green500;
  const failBarColor = theme.isDarkMode ? theme.colors.red400 : theme.colors.red500;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        minWidth: 0,
        width: '100%',
      }}
    >
      <div
        css={{
          display: 'flex',
          flex: 1,
          minWidth: 0,
          height: theme.spacing.sm,
          borderRadius: theme.borders.borderRadiusMd,
          overflow: 'hidden',
        }}
      >
        {passCount > 0 && (
          <div
            css={{
              flex: passCount,
              backgroundColor: passBarColor,
            }}
          />
        )}
        {passCount < totalCount && (
          <div
            css={{
              flex: totalCount - passCount,
              backgroundColor: failBarColor,
            }}
          />
        )}
      </div>
      <span
        css={{
          flexShrink: 0,
          fontSize: theme.typography.fontSizeSm,
          color: theme.colors.textPrimary,
          whiteSpace: 'nowrap',
        }}
      >
        <span css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
          {passCount}/{totalCount}
        </span>{' '}
        <span css={{ color: theme.colors.textSecondary }}>
          <FormattedMessage
            defaultMessage="PASS"
            description="Label for an aggregate display showing how many assessments have passed or failed"
          />
        </span>
      </span>
    </div>
  );
};

interface NumericAverageDisplayProps {
  average: number;
  count: number;
}

/**
 * Renders the average of numeric assessment values with a tooltip showing count.
 */
export const NumericAverageDisplay: React.FC<NumericAverageDisplayProps> = ({ average, count }) => {
  const { theme } = useDesignSystemTheme();

  // Format average to reasonable precision (up to 2 decimal places, trimming trailing zeros)
  const formattedAverage = Number.isInteger(average) ? average.toString() : average.toFixed(2).replace(/\.?0+$/, '');

  return (
    <Tooltip
      componentId="mlflow.genai-traces-table.session-numeric-assessment"
      content={
        <FormattedMessage
          defaultMessage="Average of {count} values"
          description="Tooltip for numeric assessment average in session header"
          values={{ count }}
        />
      }
    >
      <Tag componentId="mlflow.genai-traces-table.average-values-tag">
        <Typography.Text size="sm">
          <FormattedMessage
            defaultMessage="{average} (AVG)"
            description="Label showing the average value of numeric assessments"
            values={{ average: formattedAverage }}
          />
        </Typography.Text>
      </Tag>
    </Tooltip>
  );
};

interface StringValuesDisplayProps {
  valueCounts: Map<string, number>;
}

/**
 * Renders a tag showing unique value count with a hover card showing values and their counts.
 */
export const StringValuesDisplay: React.FC<StringValuesDisplayProps> = ({ valueCounts }) => {
  const { theme } = useDesignSystemTheme();
  const uniqueCount = valueCounts.size;

  // Sort values by count (descending)
  const sortedEntries = Array.from(valueCounts.entries()).sort((a, b) => b[1] - a[1]);

  return (
    <HoverCard
      trigger={
        <Tag css={{ cursor: 'default' }} componentId="mlflow.genai-traces-table.session-string-tag">
          <FormattedMessage
            defaultMessage="{count} {count, plural, one {value} other {values}}"
            description="Tag showing the number of unique string values in session assessment"
            values={{ count: uniqueCount }}
          />
        </Tag>
      }
      content={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          {sortedEntries.map(([value, count]) => (
            <div
              key={value}
              css={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: theme.spacing.md,
              }}
            >
              <Typography.Text ellipsis css={{ maxWidth: 200 }}>
                {value}
              </Typography.Text>
              <Typography.Text color="secondary">{count}</Typography.Text>
            </div>
          ))}
        </div>
      }
    />
  );
};
