/**
 * Shared footer component for setup wizard steps.
 */

import { Button, Spinner, useDesignSystemTheme } from '@databricks/design-system';

const COMPONENT_ID = 'mlflow.assistant.setup.footer';

interface WizardFooterProps {
  onBack?: () => void;
  onNext: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
  backDisabled?: boolean;
  isLoading?: boolean;
}

export const WizardFooter = ({
  onBack,
  onNext,
  nextLabel = 'Continue',
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
        <Button componentId={`${COMPONENT_ID}.back`} onClick={onBack} disabled={backDisabled || isLoading}>
          Back
        </Button>
      )}

      <Button componentId={`${COMPONENT_ID}.next`} type="primary" onClick={onNext} disabled={nextDisabled || isLoading}>
        {isLoading ? <Spinner size="small" /> : nextLabel}
      </Button>
    </div>
  );
};
