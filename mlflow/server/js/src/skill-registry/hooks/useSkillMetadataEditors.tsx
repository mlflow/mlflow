import { useCallback, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { useEditAliasesModal } from '../../common/hooks/useEditAliasesModal';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { diffCurrentAndNewTags } from '../../common/utils/TagUtils';
import type { KeyValueEntity } from '../../common/types';
import { SkillRegistryApi } from '../api';
import type { Skill, SkillStatus, SkillVersion } from '../types';
import { ignoreNotFound, settleAll } from '../../common/utils/registryWrites';
import { formatSkillIdentity } from '../utils';
import type { AliasMap } from '../../common/types';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';

type SkillTagEntity = { tags?: KeyValueEntity[] };

const tagsToList = (tags: Record<string, string> = {}): KeyValueEntity[] =>
  Object.entries(tags).map(([key, value]) => ({ key, value }));

export const useSkillMetadataEditors = ({
  name,
  organization,
  aliases,
  canDelete,
}: {
  name: string;
  organization: string;
  aliases: AliasMap;
  canDelete: boolean;
}) => {
  const intl = useIntl();
  const [editingTagsVersion, setEditingTagsVersion] = useState<number>();
  const invalidateQueries = useInvalidateSkillQueries();
  const invalidate = () => invalidateQueries(name, organization);

  const tagMutation = useMutation<
    unknown,
    Error,
    { version?: number; toAdd: KeyValueEntity[]; toDelete: { key: string }[] }
  >({
    mutationFn: async ({ version, toAdd, toDelete }) => {
      if (version == null) {
        await settleAll([
          ...toAdd.map(({ key, value }) => SkillRegistryApi.setSkillTag(name, { key, value }, organization)),
          ...toDelete.map(({ key }) => ignoreNotFound(SkillRegistryApi.deleteSkillTag(name, key, organization))),
        ]);
        return;
      }
      await settleAll([
        ...toAdd.map(({ key, value }) =>
          SkillRegistryApi.setSkillVersionTag(name, version, { key, value }, organization),
        ),
        ...toDelete.map(({ key }) =>
          ignoreNotFound(SkillRegistryApi.deleteSkillVersionTag(name, version, key, organization)),
        ),
      ]);
    },
    // Even a partly failed save changed the server, so the page refreshes either way.
    onSettled: () => invalidate(),
  });

  const saveTags = async (version: number | undefined, currentTags: KeyValueEntity[], newTags: KeyValueEntity[]) => {
    const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);
    // Checked up front so a save never applies the additions and then fails on the removals.
    if (deletedTags.length > 0 && !canDelete) {
      throw new Error(
        intl.formatMessage({
          defaultMessage: 'Removing tags requires the Manage permission on this skill.',
          description: 'Error when a user without delete permission removes skill tags',
        }),
      );
    }
    await tagMutation.mutateAsync({ version, toAdd: addedOrModifiedTags, toDelete: deletedTags });
  };

  const { EditTagsModal: EditSkillTagsModal, showEditTagsModal: showParentTags } =
    useEditKeyValueTagsModal<SkillTagEntity>({
      title: <FormattedMessage defaultMessage="Edit tags" description="Title for editing skill parent tags" />,
      valueRequired: true,
      saveTagsHandler: (_skill, currentTags, newTags) => saveTags(undefined, currentTags, newTags),
    });

  const { EditTagsModal: EditVersionMetadataModal, showEditTagsModal: showVersionTags } = useEditKeyValueTagsModal<{
    version: number;
    tags?: KeyValueEntity[];
  }>({
    title: (
      <FormattedMessage
        defaultMessage="Edit tags for version {version}"
        description="Title for editing tags on a skill version"
        values={{ version: editingTagsVersion }}
      />
    ),
    valueRequired: true,
    saveTagsHandler: (version, currentTags, newTags) => saveTags(version.version, currentTags, newTags),
  });

  const statusMutation = useMutation<unknown, Error, { version: number; status: SkillStatus }>({
    mutationFn: ({ version, status }) =>
      SkillRegistryApi.updateSkillVersionStatus(name, version, { status }, organization),
    onSuccess: invalidate,
  });

  const aliasMutation = useMutation<unknown, Error, { version: number; add: string[]; remove: string[] }>({
    mutationFn: async ({ version, add, remove }) => {
      await settleAll([
        ...add.map((alias) => SkillRegistryApi.setSkillAlias(name, { alias, version }, organization)),
        ...remove.map((alias) => ignoreNotFound(SkillRegistryApi.deleteSkillAlias(name, alias, organization))),
      ]);
    },
    onSettled: () => invalidate(),
  });

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases,
    getTitle: (version) => (
      <FormattedMessage
        defaultMessage="Add/Edit alias for skill version {version}"
        description="Title for editing aliases on a skill version"
        values={{ version }}
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Aliases are mutable, named pointers to a specific skill version, referenced as {uri}. The alias latest is reserved and always resolves to the skill's latest version."
        description="Description for the skill alias editor"
        values={{ uri: `skills:/${formatSkillIdentity(name, organization)}@<alias>` }}
      />
    ),
    onSave: async (version, existingAliases, draftAliases) => {
      if (draftAliases.includes('latest')) {
        throw new Error(
          intl.formatMessage({
            defaultMessage: 'The alias latest is reserved. Choose a different name.',
            description: 'Error when a user tries to set the reserved skill alias latest',
          }),
        );
      }
      const add = draftAliases.filter((alias) => !existingAliases.includes(alias));
      const remove = existingAliases.filter((alias) => alias !== 'latest' && !draftAliases.includes(alias));
      if (remove.length > 0 && !canDelete) {
        throw new Error(
          intl.formatMessage({
            defaultMessage: 'Removing aliases requires the Manage permission on this skill.',
            description: 'Error when a user without delete permission removes skill aliases',
          }),
        );
      }
      await aliasMutation.mutateAsync({ version: Number(version), add, remove });
    },
  });

  const showEditParentTags = useCallback(
    (skill: Skill) => showParentTags({ tags: tagsToList(skill.tags) }),
    [showParentTags],
  );
  const showEditVersionMetadata = useCallback(
    (version: SkillVersion) => {
      setEditingTagsVersion(version.version);
      showVersionTags({ version: version.version, tags: tagsToList(version.tags) });
    },
    [showVersionTags],
  );

  return {
    EditSkillTagsModal,
    EditVersionMetadataModal,
    EditAliasesModal,
    showEditParentTags,
    showEditVersionMetadata,
    showEditAliasesModal,
    statusMutation,
  };
};
