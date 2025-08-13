import { FolderBranchIcon, HomeIcon, NotebookIcon, WorkflowsIcon } from '@databricks/design-system';

enum SourceType {
  NOTEBOOK = 'NOTEBOOK',
  JOB = 'JOB',
  PROJECT = 'PROJECT',
  LOCAL = 'LOCAL',
  UNKNOWN = 'UNKNOWN',
}

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
