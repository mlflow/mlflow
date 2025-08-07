import { useTagAssignmentModal } from '../hooks/useTagAssignmentModal';
import type { TagAssignmentModalParams } from '../hooks/useTagAssignmentModal';

export const TagAssignmentModal = (props: TagAssignmentModalParams) => {
  const { TagAssignmentModal } = useTagAssignmentModal({ ...props });

  return TagAssignmentModal;
};
