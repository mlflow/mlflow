import { useCallback } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { useEditAliasesModal } from '../../common/hooks/useEditAliasesModal';
import { useEditKeyValueTagsModal } from '../../common/hooks/useEditKeyValueTagsModal';
import { diffCurrentAndNewTags } from '../../common/utils/TagUtils';
import type { KeyValueEntity } from '../../common/types';
import { SkillRegistryApi } from '../api';
import type { Skill, SkillVersion } from '../types';
import type { AliasMap } from '../../common/types';
import { useInvalidateSkillQueries } from './useInvalidateSkillQueries';

type SkillTagEntity = { tags?: KeyValueEntity[] };

const tagsToList = (tags: Record<string, string> = {}): KeyValueEntity[] =>
  Object.entries(tags).map(([key, value]) => ({ key, value }));

export const useSkillGovernanceModals = ({
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
  const invalidateQueries = useInvalidateSkillQueries();
  const invalidate = () => invalidateQueries(name, organization);

  const tagMutation = useMutation<
    unknown,
    Error,
    { version?: number; toAdd: KeyValueEntity[]; toDelete: { key: string }[] }
  >({
    mutationFn: async ({ version, toAdd, toDelete }) => {
      if (version == null) {
        await Promise.all([
          ...toAdd.map(({ key, value }) => SkillRegistryApi.setSkillTag(name, { key, value }, organization)),
          ...toDelete.map(({ key }) => SkillRegistryApi.deleteSkillTag(name, key, organization)),
        ]);
        return;
      }
      await Promise.all([
        ...toAdd.map(({ key, value }) =>
          SkillRegistryApi.setSkillVersionTag(name, version, { key, value }, organization),
        ),
        ...toDelete.map(({ key }) => SkillRegistryApi.deleteSkillVersionTag(name, version, key, organization)),
      ]);
    },
  });

  const saveTags = (version: number | undefined, currentTags: KeyValueEntity[], newTags: KeyValueEntity[]) => {
    const { addedOrModifiedTags, deletedTags } = diffCurrentAndNewTags(currentTags, newTags);
    // Checked up front so a save never applies the additions and then fails on the removals.
    if (deletedTags.length > 0 && !canDelete) {
      return Promise.reject(
        new Error(
          intl.formatMessage({
            defaultMessage: 'Removing tags requires the Manage permission on this skill.',
            description: 'Error when a user without delete permission removes skill tags',
          }),
        ),
      );
    }
    return new Promise<void>((resolve, reject) => {
      tagMutation.mutate(
        { version, toAdd: addedOrModifiedTags, toDelete: deletedTags },
        {
          onSuccess: () => {
            invalidate();
            resolve();
          },
          onError: reject,
        },
      );
    });
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
    title: <FormattedMessage defaultMessage="Edit metadata" description="Title for editing tags on a skill version" />,
    valueRequired: true,
    saveTagsHandler: (version, currentTags, newTags) => saveTags(version.version, currentTags, newTags),
  });

  const aliasMutation = useMutation<unknown, Error, { version: number; add: string[]; remove: string[] }>({
    mutationFn: async ({ version, add, remove }) => {
      await Promise.all([
        ...add.map((alias) => SkillRegistryApi.setSkillAlias(name, { alias, version }, organization)),
        ...remove.map((alias) => SkillRegistryApi.deleteSkillAlias(name, alias, organization)),
      ]);
    },
    onSuccess: invalidate,
  });

  const { EditAliasesModal, showEditAliasesModal } = useEditAliasesModal({
    aliases,
    getTitle: (version) => (
      <FormattedMessage
        defaultMessage="Edit aliases for version {version}"
        description="Title for editing aliases on a skill version"
        values={{ version }}
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Aliases assign a mutable name to a skill version. The name latest is reserved."
        description="Description for the skill alias editor"
      />
    ),
    onSave: async (version, existingAliases, draftAliases) => {
      const add = draftAliases.filter((alias) => alias !== 'latest' && !existingAliases.includes(alias));
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
    (version: SkillVersion) => showVersionTags({ version: version.version, tags: tagsToList(version.tags) }),
    [showVersionTags],
  );

  return {
    EditSkillTagsModal,
    EditVersionMetadataModal,
    EditAliasesModal,
    showEditParentTags,
    showEditVersionMetadata,
    showEditAliasesModal,
  };
};
