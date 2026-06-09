import type { ComponentType } from 'react';

import { z } from 'zod';
import { createComponentImplementation } from '@a2ui/react/v0_9';
import { type ComponentApi, DynamicStringSchema } from '@a2ui/web_core/v0_9';
import { CheckCircleIcon, DangerIcon, Typography, useDesignSystemTheme, XCircleFillIcon } from '@databricks/design-system';

const SENTIMENTS = ['positive', 'negative', 'neutral', 'error'] as const;
type Sentiment = (typeof SENTIMENTS)[number];

/**
 * Schema (API) for a single assessment box. The `name` (category) is the
 * header, the `value` is a colored verdict badge, and the `rationale` fills the
 * body. `sentiment` drives the color — green for positive, red for
 * negative/error, gray for neutral. Designed to be reused: a layout can append
 * one AssessmentCard per assessment into a wrapping row/board.
 */
export const AssessmentCardApi = {
  name: 'AssessmentCard',
  schema: z
    .object({
      name: DynamicStringSchema.describe('The assessment category, shown as the box header.'),
      value: DynamicStringSchema.describe('The verdict/value, shown as a colored badge.').optional(),
      rationale: DynamicStringSchema.describe('The explanation shown in the box body.').optional(),
      source: DynamicStringSchema.describe('Optional source label, e.g. "LLM_JUDGE".').optional(),
      sentiment: z
        .enum(SENTIMENTS)
        .describe('Verdict polarity: positive (green), negative/error (red), neutral (gray).')
        .default('neutral')
        .optional(),
    })
    .strict(),
} satisfies ComponentApi;

const asString = (value: unknown): string => (typeof value === 'string' ? value : String(value ?? ''));

export const AssessmentCard = createComponentImplementation(AssessmentCardApi, ({ props }) => {
  const { theme } = useDesignSystemTheme();

  const sentiment: Sentiment = (props.sentiment as Sentiment) ?? 'neutral';
  const name = asString(props.name);
  const value = props.value ? asString(props.value) : undefined;
  const rationale = props.rationale ? asString(props.rationale) : undefined;
  const source = props.source ? asString(props.source) : undefined;

  const accentBySentiment: Record<Sentiment, string> = {
    positive: theme.colors.green500,
    negative: theme.colors.red500,
    error: theme.colors.red500,
    neutral: theme.colors.border,
  };
  const verdictIconBySentiment: Record<Sentiment, ComponentType | undefined> = {
    positive: CheckCircleIcon,
    negative: XCircleFillIcon,
    error: DangerIcon,
    neutral: undefined,
  };

  const accent = accentBySentiment[sentiment];
  const VerdictIcon = verdictIconBySentiment[sentiment];

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        width: '100%',
        minWidth: 0,
        boxSizing: 'border-box',
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderLeft: `4px solid ${accent}`,
        borderRadius: theme.borders.borderRadiusMd,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: theme.spacing.sm }}>
        <Typography.Text bold size="md" css={{ minWidth: 0, overflowWrap: 'anywhere' }}>
          {name}
        </Typography.Text>
        {value && (
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              padding: `2px ${theme.spacing.sm}px`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: `${accent}1A`,
              color: accent,
              fontSize: theme.typography.fontSizeSm,
              fontWeight: 600,
              flexShrink: 0,
              maxWidth: 140,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {VerdictIcon && <VerdictIcon />}
            {value}
          </span>
        )}
      </div>
      {rationale && (
        <Typography.Text color="secondary" css={{ overflowWrap: 'anywhere' }}>
          {rationale}
        </Typography.Text>
      )}
      {source && (
        <Typography.Text color="secondary" size="sm" css={{ fontFamily: 'monospace' }}>
          {source}
        </Typography.Text>
      )}
    </div>
  );
});
