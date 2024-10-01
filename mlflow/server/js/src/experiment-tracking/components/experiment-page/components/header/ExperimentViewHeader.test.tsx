import { ExperimentViewHeader } from './ExperimentViewHeader';
import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event-14';
import { DesignSystemProvider } from '@databricks/design-system';

// mock breadcrumbs
jest.mock('@databricks/design-system', () => ({
  ...jest.requireActual('@databricks/design-system'),
  Breadcrumb: () => <div />,
}));

describe('ExperimentViewHeader', () => {
  const experiment: ExperimentEntity = {
    experimentId: '123',
    name: 'test',
    artifactLocation: 'file:/tmp/mlruns',
    lifecycleStage: 'active',
    allowedActions: [],
    creationTime: 0,
    lastUpdateTime: 0,
    tags: [],
  };

  const setEditing = (editing: boolean) => {
    return;
  };

  const createComponentMock = (showAddDescriptionButton: boolean) => {
    return renderWithIntl(
      <DesignSystemProvider>
        <ExperimentViewHeader
          experiment={experiment}
          showAddDescriptionButton={showAddDescriptionButton}
          setEditing={setEditing}
        />
      </DesignSystemProvider>,
    );
  };

  test('should render add description button', async () => {
    await act(async () => {
      createComponentMock(true);
    });

    expect(screen.queryByText('Add Description')).toBeInTheDocument();
  });

  test('should not render add description button', async () => {
    await act(async () => {
      createComponentMock(false);
    });

    expect(screen.queryByText('Add Description')).not.toBeInTheDocument();
  });
});
