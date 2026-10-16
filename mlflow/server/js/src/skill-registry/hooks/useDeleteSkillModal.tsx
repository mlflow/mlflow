import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { SkillRegistryApi } from '../api';
import type { SkillMutationResponse } from '../types';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';

export const useDeleteSkillModal = ({
  name,
  organization,
  onDeleted,
}: {
  name: string;
  organization: string;
  onDeleted: () => void;
}) => {
  const intl = useIntl();
  const invalidate = useInvalidateSkillQueries();
  const [visible, setVisible] = useState(false);
  const mutation = useMutation<SkillMutationResponse, Error, void>({
    mutationFn: () => SkillRegistryApi.deleteSkill(name, organization),
    // Not awaited: the skill is gone, so waiting on its own queries' refetch would only show "not found"
    // before the dialog navigates away.
    onSuccess: () => {
      void invalidate(name, organization);
    },
  });

  const DeleteSkillModal = (
    <ConfirmationModal
      componentId="mlflow.skill_registry.detail.delete_skill_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete skill',
        description: 'Title for confirming deletion of a skill',
      })}
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Confirm button for deleting a skill',
      })}
      visible={visible}
      message={
        <FormattedMessage
          defaultMessage="Are you sure you want to delete {name} and all of its versions? This action cannot be undone."
          description="Confirmation for hard-deleting a skill and its versions"
          values={{ name }}
        />
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        mutation.mutate(undefined, {
          onSuccess: () => {
            setVisible(false);
            onDeleted();
          },
        });
      }}
      onCancel={() => {
        mutation.reset();
        setVisible(false);
      }}
    />
  );

  return { DeleteSkillModal, openDeleteSkill: () => setVisible(true) };
};
