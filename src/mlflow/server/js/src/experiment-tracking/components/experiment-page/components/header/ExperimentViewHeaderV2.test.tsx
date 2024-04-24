import { ExperimentViewHeaderV2 } from './ExperimentViewHeaderV2';
import { renderWithIntl, act, fireEvent, screen, within } from 'common/utils/TestUtils.react18';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';
import { ExperimentEntity, KeyValueEntity } from 'experiment-tracking/types';
import { useState } from 'react';
import { it } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { Breadcrumb } from '@databricks/design-system';
import { DesignSystemProvider } from '@databricks/design-system';

// mock breadcrumbs
jest.mock('@databricks/design-system', () => ({
  ...jest.requireActual('@databricks/design-system'),
  Breadcrumb: () => <div />,
}));

describe('ExperimentViewHeaderV2', () => {
  const experiment: ExperimentEntity = {
    experiment_id: '123',
    name: 'test',
    artifact_location: 'file:/tmp/mlruns',
    lifecycle_stage: 'active',
    allowed_actions: [],
    creation_time: 0,
    last_update_time: 0,
    tags: [],
    getAllowedActions: () => [],
    getArtifactLocation: () => 'file:/tmp/mlruns',
    getCreationTime: () => 0,
    getExperimentId: () => '123',
    getLastUpdateTime: () => 0,
    getLifecycleStage: () => 'active',
    getName: () => 'test',
    getTags: () => [],
  };

  const setEditing = (editing: boolean) => {
    return;
  };

  const createComponentMock = (showAddDescriptionButton: boolean) => {
    return renderWithIntl(
      <DesignSystemProvider>
        <ExperimentViewHeaderV2
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
