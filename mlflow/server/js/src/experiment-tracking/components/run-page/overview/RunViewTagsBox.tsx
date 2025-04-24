import { Button, PencilIcon, Spinner, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { KeyValueEntity } from '../../../types';
import { KeyValueTag } from '../../../../common/components/KeyValueTag';
import { FormattedMessage, useIntl } from 'react-intl';
import { keys, values } from 'lodash';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '../../../../redux-types';
import { setRunTagsBulkApi } from '../../../actions';
import { MLFLOW_INTERNAL_PREFIX } from '../../../../common/utils/TagUtils';
import { useMemo } from 'react';
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
  const { theme } = useDesignSystemTheme();
  const dispatch = useDispatch<ThunkDispatch>();
  const intl = useIntl();

  // Get keys and tag entities while excluding system tags
  const [visibleTagKeys, visibleTagEntities] = useMemo(
    () => [keys(tags).filter(isUserFacingTag), values(tags).filter(({ key }) => isUserFacingTag(key))],
    [tags],
  );

  const { EditTagsModal, showEditTagsModal, isLoading } = useEditKeyValueTagsModal({
    valueRequired: true,
    allAvailableTags: visibleTagKeys,
    saveTagsHandler: async (_, existingTags, newTags) =>
      dispatch(setRunTagsBulkApi(runUuid, existingTags, newTags)).then(onTagsUpdated),
  });

  const showEditModal = () => {
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
      {visibleTagEntities.length < 1 ? (
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
          {visibleTagEntities.map((tag) => (
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
      {EditTagsModal}
    </div>
  );
};
