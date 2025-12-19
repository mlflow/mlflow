import { DangerIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export const ErrorCell = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <span
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        svg: { width: '12px', height: '12px' },
        color: theme.colors.textValidationWarning,
      }}
    >
      <DangerIcon css={{ color: theme.colors.textValidationWarning }} />
      <FormattedMessage defaultMessage="Error" description="Error status in the evaluations table." />{' '}
    </span>
  );
};
