import { Typography, type ThemeType } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const StepSection = ({
  theme,
  stepNumber,
  title,
  description,
  children,
  isPending,
}: {
  theme: ThemeType;
  stepNumber: number;
  title: React.ReactNode;
  description: React.ReactNode;
  children?: React.ReactNode;
  isPending?: boolean;
}) => {
  return (
    <div css={{ display: 'flex', gap: theme.spacing.md }}>
      {/* Step indicator */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: 2,
        }}
      >
        <div
          css={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: isPending ? 'transparent' : theme.colors.actionPrimaryBackgroundDefault,
            border: isPending ? `2px dashed ${theme.colors.border}` : 'none',
            color: isPending ? theme.colors.textSecondary : '#fff',
            fontSize: 13,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          {stepNumber}
        </div>
        {!isPending && (
          <div
            css={{
              width: 2,
              flex: 1,
              backgroundColor: theme.colors.border,
              marginTop: theme.spacing.xs,
            }}
          />
        )}
      </div>

      {/* Content */}
      <div css={{ flex: 1, minWidth: 0, paddingBottom: isPending ? 0 : theme.spacing.sm }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
          <Typography.Title level={4} withoutMargins>
            {title}
          </Typography.Title>
          {isPending && (
            <span
              css={{
                fontSize: 11,
                fontWeight: 600,
                color: theme.colors.textValidationWarning,
                backgroundColor: `${theme.colors.textValidationWarning}15`,
                padding: '2px 8px',
                borderRadius: 10,
              }}
            >
              <FormattedMessage
                defaultMessage="Pending"
                description="Label indicating this onboarding step is not yet completed"
              />
            </span>
          )}
        </div>
        <Typography.Paragraph color="secondary" css={{ marginBottom: children ? theme.spacing.sm : 0 }}>
          {description}
        </Typography.Paragraph>
        {children}
      </div>
    </div>
  );
};
