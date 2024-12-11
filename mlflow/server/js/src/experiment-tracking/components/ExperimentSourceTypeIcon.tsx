import { FolderBranchIcon, HomeIcon, NotebookIcon, WorkflowsIcon } from '@databricks/design-system';
import { SourceType } from '../sdk/MlflowEnums';

/**
 * Displays an icon corresponding to the source type of an experiment run.
 */
export const ExperimentSourceTypeIcon = ({
  sourceType,
  className,
}: {
  sourceType: SourceType | string;
  className?: string;
}) => {
  if (sourceType === SourceType.NOTEBOOK) {
    return <NotebookIcon className={className} />;
  } else if (sourceType === SourceType.LOCAL) {
    return <HomeIcon className={className} />;
  } else if (sourceType === SourceType.PROJECT) {
    return <FolderBranchIcon className={className} />;
  } else if (sourceType === SourceType.JOB) {
    return <WorkflowsIcon className={className} />;
  }
  return null;
};
