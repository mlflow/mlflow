import {
  Button,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxFooter,
  PlusIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState } from 'react';
import { isUserFacingTag } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { setRunTagsBulkApi } from '@mlflow/mlflow/src/experiment-tracking/actions';
import type { RunInfoEntity } from '../../../../types';
import type { KeyValueEntity } from '../../../../../common/types';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import { ExperimentViewRunsControlsActionsAddNewTagModal } from './ExperimentViewRunsControlsActionsAddNewTagModal';
import { uniq } from 'lodash';
import { FormattedMessage } from 'react-intl';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { ErrorWrapper } from '@mlflow/mlflow/src/common/utils/ErrorWrapper';

const convertTagToString = (tag: KeyValueEntity) => {
  return `${tag.key}: ${tag.value}`;
};
const convertStringToTag = (tagString: string) => {
  const sep = ': ';
  const [key, ...splits] = tagString.split(sep);
  return { key, value: splits.join(sep) };
};

const getRunsTagsSelection = (
  runInfos: RunInfoEntity[],
  runsSelected: Record<string, boolean>,
  tagsList: Record<string, KeyValueEntity>[],
) => {
  const selectedRunsTagArray: string[][] = runInfos.flatMap((run, idx) => {
    if (runsSelected[run.runUuid]) {
      const tags = tagsList[idx];
      return [
        Object.keys(tags)
          .filter(isUserFacingTag)
          .map((tagKey) => convertTagToString(tags[tagKey])),
      ];
    }
    return [];
  });

  const allRunsTags: string[] = tagsList.flatMap((tags) => {
    return Object.keys(tags)
      .filter(isUserFacingTag)
      .map((tagKey) => convertTagToString(tags[tagKey]));
  });

  const selectedRunsAllSelectedTags: string[] = allRunsTags.filter((tag) =>
    selectedRunsTagArray.every((selectedTags) => selectedTags.includes(tag)),
  );
  const selectedRunsAllNotSelectedTags: string[] = allRunsTags.filter((tag) =>
    selectedRunsTagArray.every((selectedTags) => !selectedTags.includes(tag)),
  );
  const selectedRunsIndeterminateTags: string[] = allRunsTags.filter(
    (tag) =>
      !selectedRunsAllSelectedTags.includes(tag) &&
      selectedRunsTagArray.some((selectedTags) => selectedTags.includes(tag)),
  );

  return {
    allSelectedTags: selectedRunsAllSelectedTags,
    allNotSelectedTags: selectedRunsAllNotSelectedTags,
    indeterminateTags: selectedRunsIndeterminateTags,
    allTags: allRunsTags,
  };
};

export const ExperimentViewRunsControlsActionsSelectTags = ({
  runInfos,
  runsSelected,
  tagsList,
  refreshRuns,
}: {
  runInfos: RunInfoEntity[];
  runsSelected: Record<string, boolean>;
  tagsList: Record<string, KeyValueEntity>[];
  refreshRuns: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [selectedTags, setSelectedTags] = useState<Record<string, boolean | undefined>>({});
  const [isAddNewTagModalOpen, setIsAddNewTagModalOpen] = useState(false);
  const [isMultiSelectOpen, setIsMultiSelectOpen] = useState(false);
  const [isSavingTagsLoading, setIsSavingTagsLoading] = useState(false);

  const { allSelectedTags, allNotSelectedTags, indeterminateTags, allTags } = getRunsTagsSelection(
    runInfos,
    runsSelected,
    tagsList,
  );

  const openDropdown = (newTag?: KeyValueEntity) => {
    setSelectedTags(() => {
      const selectedValues: Record<string, boolean | undefined> = { ...selectedTags };
      allTags.forEach((tag) => {
        if (allSelectedTags.includes(tag)) {
          selectedValues[tag] = true;
        } else if (allNotSelectedTags.includes(tag)) {
          selectedValues[tag] = false;
        } else if (indeterminateTags.includes(tag)) {
          selectedValues[tag] = undefined;
        }
      });
      if (newTag !== undefined) {
        selectedValues[convertTagToString(newTag)] = true;
      }
      return selectedValues;
    });
    setIsMultiSelectOpen(true);
  };

  const handleChange = (updatedTagString: string) => {
    setSelectedTags((selectedTags) => ({
      ...selectedTags,
      [updatedTagString]: !selectedTags[updatedTagString],
    }));
  };

  const dispatch = useDispatch<ThunkDispatch>();

  const saveTags = () => {
    setIsSavingTagsLoading(true);
    const selectedRunIdxs = runInfos.flatMap((runInfo, idx) => (runsSelected[runInfo.runUuid] ? [idx] : []));
    selectedRunIdxs.forEach((idx) => {
      const runUuid = runInfos[idx].runUuid;
      // Get all non-system tags for the selected run
      const existingKeys = Object.values(tagsList[idx]).filter((tag) => isUserFacingTag(tag.key));
      // Get all new tags that are explicitly selected. If its indeterminate, and the key is in existingKeys, then it should stay
      const newKeys = Object.keys(selectedTags)
        .filter((tag) => {
          if (selectedTags[tag] === undefined) {
            return existingKeys.map((tag) => convertTagToString(tag)).includes(tag);
          } else {
            return selectedTags[tag];
          }
        })
        .map((tagString) => convertStringToTag(tagString));
      dispatch(setRunTagsBulkApi(runUuid, existingKeys, newKeys))
        .then(() => {
          refreshRuns();
        })
        .catch((e) => {
          const message = e instanceof ErrorWrapper ? e.getMessageField() : e.message;
          Utils.displayGlobalErrorNotification(message);
        })
        .finally(() => {
          setIsSavingTagsLoading(false);
          setIsMultiSelectOpen(false);
        });
    });
  };

  const addNewTag = (tag: KeyValueEntity) => {
    openDropdown(tag);
  };

  const addNewTagModal = () => {
    setIsAddNewTagModalOpen(true);
    setIsMultiSelectOpen(false);
  };

  return (
    <>
      <DialogCombobox
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunscontrolsactionsselecttags.tsx_162"
        open={isMultiSelectOpen}
        label="Add tags"
        id="runs-tag-multiselect"
        multiSelect
      >
        <DialogComboboxTrigger
          onClick={() => {
            if (isMultiSelectOpen) {
              setIsMultiSelectOpen(false);
            } else {
              // Open the dropdown and render tag selections.
              openDropdown();
            }
          }}
          data-testid="runs-tag-multiselect-trigger"
        />
        <DialogComboboxContent matchTriggerWidth>
          <DialogComboboxOptionList>
            {Object.keys(selectedTags).map((tagString) => {
              const isIndeterminate = selectedTags[tagString] === undefined;
              return (
                <DialogComboboxOptionListCheckboxItem
                  key={tagString}
                  value={tagString}
                  onChange={handleChange}
                  checked={selectedTags[tagString]}
                  indeterminate={isIndeterminate}
                />
              );
            })}
          </DialogComboboxOptionList>
          <DialogComboboxFooter>
            <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
              <Button
                componentId="mlflow.experiment_page.runs.add_new_tag"
                onClick={addNewTagModal}
                icon={<PlusIcon />}
                data-testid="runs-add-new-tag-button"
              >
                <FormattedMessage
                  defaultMessage="Add new tag"
                  description="Experiment tracking > experiment page > runs > add new tag button"
                />
              </Button>
              <Button
                type="primary"
                componentId="mlflow.experiment_page.runs.add_tags"
                onClick={saveTags}
                disabled={Object.keys(selectedTags).length === 0}
                loading={isSavingTagsLoading}
              >
                <FormattedMessage
                  defaultMessage="Save"
                  description="Experiment tracking > experiment page > runs > save tags button"
                />
              </Button>
            </div>
          </DialogComboboxFooter>
        </DialogComboboxContent>
      </DialogCombobox>
      <ExperimentViewRunsControlsActionsAddNewTagModal
        isOpen={isAddNewTagModalOpen}
        setIsOpen={setIsAddNewTagModalOpen}
        selectedRunsExistingTagKeys={uniq(
          allSelectedTags.concat(indeterminateTags).map((tag) => convertStringToTag(tag).key),
        )}
        addNewTag={addNewTag}
      />
    </>
  );
};
