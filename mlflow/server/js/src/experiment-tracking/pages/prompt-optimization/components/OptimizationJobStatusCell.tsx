import type { ColumnDef } from '@tanstack/react-table';
import {
  CheckCircleIcon,
  ClockIcon,
  Spinner,
  XCircleIcon,
  MinusCircleIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { PromptOptimizationJob } from '../types';
import { JobStatus, OptimizerType, getJobStatusName, getJobProgress } from '../types';
import { Progress } from '@mlflow/mlflow/src/common/components/Progress';

export const OptimizationJobStatusCell: ColumnDef<PromptOptimizationJob>['cell'] = ({ row: { original } }) => {
  const { theme } = useDesignSystemTheme();
  const status = original.state?.status;
  const isGEPA = original.config?.optimizer_type === OptimizerType.GEPA;
  const progress = getJobProgress(original);

  const getStatusIcon = () => {
    switch (status) {
      case JobStatus.COMPLETED:
        return <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />;
      case JobStatus.FAILED:
        return <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />;
      case JobStatus.CANCELED:
        return <MinusCircleIcon css={{ color: theme.colors.textSecondary }} />;
      case JobStatus.IN_PROGRESS:
        return <Spinner size="small" />;
      case JobStatus.PENDING:
        return <ClockIcon css={{ color: theme.colors.textSecondary }} />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case JobStatus.COMPLETED:
        return theme.colors.textValidationSuccess;
      case JobStatus.FAILED:
        return theme.colors.textValidationDanger;
      case JobStatus.CANCELED:
        return theme.colors.textSecondary;
      case JobStatus.IN_PROGRESS:
        return theme.colors.textValidationWarning;
      case JobStatus.PENDING:
        return theme.colors.textSecondary;
      default:
        return theme.colors.textPrimary;
    }
  };

  // Show progress bar for GEPA jobs that are in progress
  const showProgressBar = isGEPA && status === JobStatus.IN_PROGRESS && progress !== undefined;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, minWidth: 120 }}>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        {getStatusIcon()}
        <span css={{ color: getStatusColor() }}>{getJobStatusName(status)}</span>
      </div>
      {showProgressBar && (
        <Progress percent={Math.round((progress ?? 0) * 100)} format={(p) => `${p}%`} css={{ width: '100%' }} />
      )}
    </div>
  );
};
