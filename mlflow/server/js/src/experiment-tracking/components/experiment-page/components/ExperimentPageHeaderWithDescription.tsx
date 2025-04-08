import { useMemo, useState } from 'react';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { ExperimentViewHeader, ExperimentViewHeaderSkeleton } from './header/ExperimentViewHeader';
import { ExperimentEntity } from '../../../types';
import { ExperimentViewDescriptionNotes } from './ExperimentViewDescriptionNotes';
import { NOTE_CONTENT_TAG } from '../../../utils/NoteUtils';
import { ApolloError } from '@mlflow/mlflow/src/common/utils/graphQLHooks';
import { getGraphQLErrorMessage } from '../../../../graphql/get-graphql-error';
import { Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

type GetExperimentReturnType = ReturnType<typeof useGetExperimentQuery>['data'];

/**
 * Renders experiment page header with description and notes editor.
 */
export const ExperimentPageHeaderWithDescription = ({
  experiment,
  loading,
  onNoteUpdated,
  error,
}: {
  experiment: GetExperimentReturnType;
  loading?: boolean;
  onNoteUpdated?: () => void;
  error: ApolloError | ReturnType<typeof useGetExperimentQuery>['apiError'];
}) => {
  const { theme } = useDesignSystemTheme();
  const [showAddDescriptionButton, setShowAddDescriptionButton] = useState(true);
  const [editing, setEditing] = useState(false);

  // Coerce experiment from the query result into the shape expected by <ExperimentViewHeader />
  const experimentEntity = useMemo(() => {
    const experimentResponse = experiment as GetExperimentReturnType;
    if (!experimentResponse) return null;
    return {
      ...experimentResponse,
      creationTime: Number(experimentResponse?.creationTime),
      lastUpdateTime: Number(experimentResponse?.lastUpdateTime),
    } as ExperimentEntity;
  }, [experiment]);

  const experimentDescription = experimentEntity?.tags?.find((tag) => tag.key === NOTE_CONTENT_TAG)?.value;
  const errorMessage = getGraphQLErrorMessage(error);

  if (loading) {
    return <ExperimentViewHeaderSkeleton />;
  }

  if (errorMessage) {
    return (
      <div css={{ height: theme.general.heightBase, marginTop: theme.spacing.sm, marginBottom: theme.spacing.md }}>
        <Alert
          componentId="mlflow.logged_model.list.header.error"
          type="error"
          message={
            <FormattedMessage
              defaultMessage="Experiment load error: {errorMessage}"
              description="Error message displayed on logged models page when experiment data fails to load"
              values={{ errorMessage }}
            />
          }
          closable={false}
        />
      </div>
    );
  }

  if (experimentEntity) {
    return (
      <>
        <ExperimentViewHeader
          experiment={experimentEntity}
          showAddDescriptionButton={showAddDescriptionButton}
          setEditing={setEditing}
        />
        <ExperimentViewDescriptionNotes
          experiment={experimentEntity}
          setShowAddDescriptionButton={setShowAddDescriptionButton}
          editing={editing}
          setEditing={setEditing}
          onNoteUpdated={onNoteUpdated}
          defaultValue={experimentDescription}
        />
      </>
    );
  }

  return null;
};
