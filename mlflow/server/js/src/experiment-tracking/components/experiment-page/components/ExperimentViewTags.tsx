import { FormattedMessage } from 'react-intl';
import React from 'react';
import { useSelector } from 'react-redux';
import { CollapsibleSection } from '../../../../common/components/CollapsibleSection';
import { EditableTagsTableView } from '../../../../common/components/EditableTagsTableView';
import { ExperimentEntity } from '../../../types';
import { getExperimentTags } from '../../../reducers/Reducers';
import { useFetchExperiments } from '../hooks/useFetchExperiments';

export interface ExperimentViewTagsProps {
  experiment: ExperimentEntity;
}

/**
 * ExperimentView part responsible for displaying/editing tags.
 *
 * Consumes tags from the redux store and dispatches
 * `setExperimentTagApi` redux action from the context.
 */
export const ExperimentViewTags = React.memo(({ experiment }: ExperimentViewTagsProps) => {
  const storedTags = useSelector((state) => {
    const tags = getExperimentTags(experiment.experiment_id, state);
    return tags;
  });

  const formRef = React.createRef();

  const {
    actions: { setExperimentTagApi, deleteExperimentTagApi },
  } = useFetchExperiments();

  const handleSaveEdit = ({ name, value }: { name: string; value: string }) => {
    const action = setExperimentTagApi(experiment.experiment_id, name, value).payload.catch(
      (ex) => {
        console.error(ex);
      },
    );
    return action;
  };

  const handleDeleteTag = ({ name }: { name: string }) => {
    const action = deleteExperimentTagApi(experiment.experiment_id, name).payload.catch((ex) => {
      console.error(ex);
    });
    return action;
  };

  return (
    <CollapsibleSection
      title={
        <span css={styles.collapsibleSectionHeader}>
          <FormattedMessage
            defaultMessage='Tags'
            description='Header for displaying tags for the experiment table'
          />{' '}
        </span>
      }
      defaultCollapsed
      data-test-id='experiment-tags-section'
    >
      <EditableTagsTableView
        innerRef={formRef}
        handleAddTag={null}
        handleDeleteTag={handleDeleteTag}
        handleSaveEdit={handleSaveEdit}
        tags={storedTags}
        isRequestPending={false}
      />
    </CollapsibleSection>
  );
});

const styles = {
  collapsibleSectionHeader: {
    height: '32px',
    lineHeight: '32px',
  },
};
