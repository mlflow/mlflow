import { CheckCircleIcon, ClockIcon, XCircleIcon, useDesignSystemTheme } from '@databricks/design-system';

const ErrorIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
};

const FinishedIcon = () => {
  const { theme } = useDesignSystemTheme();
  return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
};

export const RunStatusIcon = ({ status }: { status: string }) => {
  switch (status) {
    case 'FAILED':
    case 'KILLED':
      return <ErrorIcon />;
    case 'FINISHED':
      return <FinishedIcon />;
    case 'SCHEDULED':
    case 'RUNNING':
      return <ClockIcon />; // This one is the same color as the link
    default:
      return null;
  }
};
