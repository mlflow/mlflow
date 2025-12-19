import { PencilIcon, TrashIcon, OverflowIcon, Button, DropdownMenu } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';

import type { Assessment } from '../ModelTrace.types';

export const AssessmentActionsOverflowMenu = ({
  assessment,
  setIsEditing,
  setShowDeleteModal,
}: {
  assessment: Assessment;
  setIsEditing?: (isEditing: boolean) => void;
  setShowDeleteModal: (showDeleteModal: boolean) => void;
}) => {
  const isFeedback = 'feedback' in assessment;
  const user = getUser() ?? '';

  const doesUserHavePermissions =
    user === assessment.source.source_id || (isFeedback && assessment.source.source_type !== 'HUMAN');
  const showEditButton = doesUserHavePermissions && setIsEditing;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button componentId="shared.model-trace-explorer.assessment-more-button" icon={<OverflowIcon />} size="small" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Content minWidth={100}>
        {showEditButton && (
          <DropdownMenu.Item
            componentId="shared.model-trace-explorer.assessment-edit-button"
            onClick={() => setIsEditing?.(true)}
          >
            <DropdownMenu.IconWrapper>
              <PencilIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Edit" description="Edit assessment menu item" />
          </DropdownMenu.Item>
        )}
        <DropdownMenu.Item
          componentId="shared.model-trace-explorer.assessment-delete-button"
          onClick={() => setShowDeleteModal(true)}
        >
          <DropdownMenu.IconWrapper>
            <TrashIcon />
          </DropdownMenu.IconWrapper>
          <FormattedMessage defaultMessage="Delete" description="Delete assessment menu item" />
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};
