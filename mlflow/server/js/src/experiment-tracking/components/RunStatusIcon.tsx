import { CheckCircleIcon, ClockIcon, Spinner, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';

const ErrorIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
};

const FinishedIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
};

export const RunStatusIcon = ({ status, useSpinner }: { status: string; useSpinner?: boolean }) => {
  switch (status) {
    case 'FAILED':
    case 'KILLED':
      return <ErrorIcon />;
    case 'FINISHED':
      return <FinishedIcon />;
    case 'SCHEDULED':
    case 'RUNNING':
      return useSpinner ? <Spinner size="small" /> : <ClockIcon />;
    default:
      return null;
  }
};
