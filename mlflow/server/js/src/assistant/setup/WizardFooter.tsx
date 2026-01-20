/**
 * Shared footer component for setup wizard steps.
 */

import { Button, Spinner, useDesignSystemTheme } from '@databricks/design-system';

interface WizardFooterProps {
  onBack?: () => void;
  onNext: () => void;
  nextLabel?: string;
  backLabel?: string;
  nextDisabled?: boolean;
  backDisabled?: boolean;
  isLoading?: boolean;
}

export const WizardFooter = ({
  onBack,
  onNext,
  nextLabel = 'Continue',
  backLabel = 'Back',
  nextDisabled = false,
  backDisabled = false,
  isLoading = false,
}: WizardFooterProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        justifyContent: onBack ? 'space-between' : 'flex-end',
        marginTop: theme.spacing.lg,
        paddingTop: theme.spacing.md,
        borderTop: `1px solid ${theme.colors.border}`,
      }}
    >
      {onBack && (
        <Button componentId="mlflow.assistant.setup.footer.back" onClick={onBack} disabled={backDisabled || isLoading}>
          Back
        </Button>
      )}

      <Button componentId="mlflow.assistant.setup.footer.next" type="primary" onClick={onNext} disabled={nextDisabled || isLoading}>
        {isLoading ? <Spinner size="small" /> : nextLabel}
      </Button>
    </div>
  );
};
