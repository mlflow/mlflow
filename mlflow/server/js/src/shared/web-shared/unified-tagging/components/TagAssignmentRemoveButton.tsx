import type { ButtonProps } from '@databricks/design-system';

import { TagAssignmentRemoveButtonUI } from './TagAssignmentUI/TagAssignmentRemoveButtonUI';
import { useTagAssignmentContext } from '../context/TagAssignmentContextProvider';

export interface TagAssignmentRemoveButtonProps extends Omit<ButtonProps, 'onClick' | 'icon'> {
  index: number;
}

export function TagAssignmentRemoveButton({ index, ...props }: TagAssignmentRemoveButtonProps) {
  const { removeOrUpdate } = useTagAssignmentContext();

  return <TagAssignmentRemoveButtonUI onClick={() => removeOrUpdate(index)} {...props} />;
}
