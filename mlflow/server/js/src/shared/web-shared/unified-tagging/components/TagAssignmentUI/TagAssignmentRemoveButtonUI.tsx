import type { ButtonProps } from '@databricks/design-system';
import { Button, TrashIcon } from '@databricks/design-system';

export function TagAssignmentRemoveButtonUI(props: Omit<ButtonProps, 'icon'>) {
  return <Button icon={<TrashIcon />} {...props} />;
}
