import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { GuardrailStage } from '../../types';

export const PipelineStagePicker = ({
  stage,
  onChange,
}: {
  stage: GuardrailStage;
  onChange: (s: GuardrailStage) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div>
      <Typography.Text
        bold
        css={{ display: 'block', fontSize: theme.typography.fontSizeLg, marginBottom: theme.spacing.xs }}
      >
        <FormattedMessage defaultMessage="Stage" description="Guardrail stage label" />
      </Typography.Text>
      <Typography.Text
        color="secondary"
        css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}
      >
        <FormattedMessage
          defaultMessage="Click on a stage to choose where this guardrail runs."
          description="Stage help text"
        />
      </Typography.Text>
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.borders.borderRadiusMd,
        }}
      >
        {(['Request', 'BEFORE', 'LLM', 'AFTER', 'Response'] as const).map((item, i) => {
          const isStage = item === 'BEFORE' || item === 'AFTER';
          const isSelected = isStage && item === stage;
          const label =
            item === 'BEFORE'
              ? intl.formatMessage({ defaultMessage: 'Before Guardrails', description: 'Pipeline BEFORE stage label' })
              : item === 'AFTER'
                ? intl.formatMessage({ defaultMessage: 'After Guardrails', description: 'Pipeline AFTER stage label' })
                : item;
          return (
            <div key={item} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              {i > 0 && (
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                  {'>'}
                </Typography.Text>
              )}
              {isStage ? (
                <div
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => onChange(item as GuardrailStage)}
                  onKeyDown={(e) => e.key === 'Enter' && onChange(item as GuardrailStage)}
                  tabIndex={0}
                  css={{
                    padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                    borderRadius: theme.borders.borderRadiusMd,
                    border: `1.5px dashed ${isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
                    cursor: 'pointer',
                    fontWeight: isSelected ? theme.typography.typographyBoldFontWeight : 'normal',
                    color: isSelected ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.textPrimary,
                    whiteSpace: 'nowrap',
                    userSelect: 'none',
                    '&:hover': {
                      borderColor: theme.colors.actionPrimaryBackgroundDefault,
                      color: theme.colors.actionPrimaryBackgroundDefault,
                    },
                  }}
                >
                  {label}
                </div>
              ) : (
                <Typography.Text
                  color="secondary"
                  css={{ fontSize: theme.typography.fontSizeSm, whiteSpace: 'nowrap' }}
                >
                  {label}
                </Typography.Text>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
