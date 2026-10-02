import { useState } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { ConfirmationModal } from '../../admin/ConfirmationModal';
import { SkillRegistryApi } from '../api';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';

export const useDeleteSkillVersionModal = ({
  name,
  organization,
  onDeleted,
}: {
  name: string;
  organization: string;
  onDeleted: (version: number) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const invalidate = useInvalidateSkillQueries();
  const [versionToDelete, setVersionToDelete] = useState<number | undefined>();
  const mutation = useMutation<unknown, Error, number>({
    mutationFn: (version) => SkillRegistryApi.deleteSkillVersion(name, version, organization),
    onSuccess: (_result, version) => {
      invalidate(name, organization);
      onDeleted(version);
    },
  });

  const DeleteSkillVersionModal = (
    <ConfirmationModal
      componentId="mlflow.skill_registry.detail.delete_version_modal"
      title={intl.formatMessage({
        defaultMessage: 'Delete version',
        description: 'Title for confirming skill version deletion',
      })}
      okText={intl.formatMessage({
        defaultMessage: 'Delete',
        description: 'Confirm button for deleting a skill version',
      })}
      visible={versionToDelete != null}
      message={
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Text>
            <FormattedMessage
              defaultMessage="Delete version {version}? Its number is retained and never reused, but the version is removed from resolution, discovery and pull, the registry stops returning it, and any alias pointing at it is dropped. This cannot be undone."
              description="Confirmation for soft-deleting a skill version"
              values={{ version: versionToDelete ?? '' }}
            />
          </Typography.Text>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Retiring this version rather than withdrawing it? Deprecate it instead — a deprecated version still resolves for anything that pins it, so consumers keep working."
              description="Hint that deprecating a skill version is less disruptive than deleting it"
            />
          </Typography.Text>
        </div>
      }
      isLoading={mutation.isLoading}
      error={mutation.error?.message ?? null}
      onConfirm={() => {
        if (versionToDelete != null) {
          mutation.mutate(versionToDelete, { onSuccess: () => setVersionToDelete(undefined) });
        }
      }}
      onCancel={() => {
        mutation.reset();
        setVersionToDelete(undefined);
      }}
    />
  );

  return { DeleteSkillVersionModal, openDeleteSkillVersion: (version: number) => setVersionToDelete(version) };
};
