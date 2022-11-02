import { Button } from '@databricks/design-system';
import React, { useCallback, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { useSelector } from 'react-redux';
import { CollapsibleSection } from '../../../../common/components/CollapsibleSection';
import { EditableNote } from '../../../../common/components/EditableNote';
import { getExperimentTags } from '../../../reducers/Reducers';
import { ExperimentEntity, KeyValueEntity } from '../../../types';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import { useAsyncDispatch } from '../hooks/useAsyncDispatch';
import { useFetchExperiments } from '../hooks/useFetchExperiments';

const extractNoteFromTags = (tags: Record<string, KeyValueEntity>) =>
  Object.values(tags).find((t) => t.getKey() === NOTE_CONTENT_TAG)?.value || undefined;

export interface ExperimentViewNotesProps {
  experiment: ExperimentEntity;
}

/**
 * ExperimentView part responsible for displaying/editing note.
 *
 * Consumes note from the redux store and dispatches
 * `setExperimentTagApi` redux action from the context.
 */
export const ExperimentViewNotes = React.memo(({ experiment }: ExperimentViewNotesProps) => {
  const storedNote = useSelector((state) => {
    const tags = getExperimentTags(experiment.experiment_id, state);
    return tags ? extractNoteFromTags(tags) : '';
  });

  const [showNotesEditor, setShowNotesEditor] = useState(false);

  const {
    actions: { setExperimentTagApi },
  } = useFetchExperiments();

  const dispatch = useAsyncDispatch();

  const handleSubmitEditNote = useCallback(
    (updatedNote: any) => {
      const action = setExperimentTagApi(experiment.experiment_id, NOTE_CONTENT_TAG, updatedNote);
      dispatch(action).then(() => setShowNotesEditor(false));
    },
    [experiment.experiment_id, setExperimentTagApi, dispatch],
  );

  return (
    <CollapsibleSection
      title={
        <span css={styles.collapsibleSectionHeader}>
          <FormattedMessage
            defaultMessage='Description'
            description='Header for displaying notes for the experiment table'
          />{' '}
          {!showNotesEditor && (
            <Button type='link' onClick={() => setShowNotesEditor(true)}>
              <FormattedMessage
                defaultMessage='Edit'
                // eslint-disable-next-line max-len
                description='Text for the edit button next to the description section title on the experiment view page'
              />
            </Button>
          )}
        </span>
      }
      forceOpen={showNotesEditor}
      defaultCollapsed={!storedNote}
      data-test-id='experiment-notes-section'
    >
      <EditableNote
        defaultMarkdown={storedNote}
        onSubmit={handleSubmitEditNote}
        onCancel={() => setShowNotesEditor(false)}
        showEditor={showNotesEditor}
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
