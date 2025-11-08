import { Button, PencilIcon, Spinner, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { shouldUseSharedTaggingUI } from '../../../../common/utils/FeatureUtils';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { useTagAssignmentModal } from '../../../../common/hooks/useTagAssignmentModal';
import type { KeyValueEntity } from '../../../../common/types';
import { KeyValueTag } from '../../../../common/components/KeyValueTag';
import { FormattedMessage, useIntl } from 'react-intl';
import { keys, values } from 'lodash';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../redux-types';
import { setRunTagsBulkApi, saveRunTagsApi } from '../../../actions';
import { useMemo, useState } from 'react';
import { isUserFacingTag } from '../../../../common/utils/TagUtils';

/**
 * Displays run tags cell in run detail overview.
 */
export const RunViewTagsBox = ({
  runUuid,
  tags,
  onTagsUpdated,
}: {
  runUuid: string;
  tags: Record<string, KeyValueEntity>;
  onTagsUpdated: () => void;
}) => {
  const sharedTaggingUIEnabled = shouldUseSharedTaggingUI();

  const [isSavingTags, setIsSavingTags] = useState(false);

  const { theme } = useDesignSystemTheme();
  const dispatch = useDispatch<ThunkDispatch>();
  const intl = useIntl();

  // Get keys and tag entities while excluding system tags
  const [visibleTagKeys, visibleTagEntities] = useMemo(
    () => [keys(tags).filter(isUserFacingTag), values(tags).filter(({ key }) => isUserFacingTag(key))],
    [tags],
  );

  const tagsKeyValueMap: KeyValueEntity[] = visibleTagEntities.map(({ key, value }) => ({ key, value }));

  const { TagAssignmentModal, showTagAssignmentModal } = useTagAssignmentModal({
    componentIdPrefix: 'mlflow.run-view-tags-box',
    initialTags: tagsKeyValueMap,
    isLoading: isSavingTags,
    onSubmit: (newTags: KeyValueEntity[], deletedTags: KeyValueEntity[]) => {
      setIsSavingTags(true);
      return dispatch(saveRunTagsApi(runUuid, newTags, deletedTags)).then(() => {
        setIsSavingTags(false);
      });
    },
    onSuccess: onTagsUpdated,
  });

  const { EditTagsModal, showEditTagsModal, isLoading } = useEditKeyValueTagsModal({
    valueRequired: true,
    allAvailableTags: visibleTagKeys,
    saveTagsHandler: async (_, existingTags, newTags) =>
      dispatch(setRunTagsBulkApi(runUuid, existingTags, newTags)).then(onTagsUpdated),
  });

  const showEditModal = () => {
    if (sharedTaggingUIEnabled) {
      showTagAssignmentModal();
      return;
    }

    showEditTagsModal({ tags: visibleTagEntities });
  };

  const editTagsLabel = intl.formatMessage({
    defaultMessage: 'Edit tags',
    description: "Run page > Overview > Tags cell > 'Edit' button label",
  });

  return (
    <div
      css={{
        paddingTop: theme.spacing.xs,
        paddingBottom: theme.spacing.xs,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        '> *': {
          marginRight: '0 !important',
        },
        gap: theme.spacing.xs,
      }}
    >
      {tagsKeyValueMap.length < 1 ? (
        <Button
          componentId="mlflow.run_details.overview.tags.add_button"
          size="small"
          type="tertiary"
          onClick={showEditModal}
        >
          <FormattedMessage
            defaultMessage="Add tags"
            description="Run page > Overview > Tags cell > 'Add' button label"
          />
        </Button>
      ) : (
        <>
          {tagsKeyValueMap.map((tag) => (
            <KeyValueTag tag={tag} key={`${tag.key}-${tag.value}`} enableFullViewModal css={{ marginRight: 0 }} />
          ))}
          <Tooltip componentId="mlflow.run_details.overview.tags.edit_button.tooltip" content={editTagsLabel}>
            <Button
              componentId="mlflow.run_details.overview.tags.edit_button"
              aria-label={editTagsLabel}
              size="small"
              icon={<PencilIcon />}
              onClick={showEditModal}
            />
          </Tooltip>
        </>
      )}
      {isLoading && <Spinner size="small" />}
      {/** Old modal for editing tags */}
      {EditTagsModal}
      {/** New modal for editing tags, using shared tagging UI */}
      {TagAssignmentModal}
    </div>
  );
};
