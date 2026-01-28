import type { ColumnDef } from '@tanstack/react-table';
import type { PromptOptimizationJob } from '../types';

/**
 * Shortens a job ID for display. Takes the last 8 characters.
 * Example: "abc123def456ghi789" -> "...hi789"
 */
const shortenJobId = (jobId: string): string => {
  if (jobId.length <= 8) {
    return jobId;
  }
  return `...${jobId.slice(-8)}`;
};

export const OptimizationJobNameCell: ColumnDef<PromptOptimizationJob>['cell'] = ({ getValue }) => {
  const jobId = getValue<string>();

  if (!jobId) {
    return <span>-</span>;
  }

  return <span>{shortenJobId(jobId)}</span>;
};
