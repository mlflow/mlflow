import { render, screen, waitFor } from '@testing-library/react';
import { ExperimentLoggedModelSourceBox } from './ExperimentLoggedModelSourceBox';
import type { LoggedModelProto } from '../../types';
import { DesignSystemProvider } from '@databricks/design-system';
import { TestRouter, testRoute } from '../../../common/utils/RoutingTestUtils';

const defaultTestTags = [
  { key: 'mlflow.source.git.branch', value: 'branch-abc' },
  { key: 'mlflow.source.git.commit', value: 'abc123def456' },
  { key: 'mlflow.source.type', value: 'NOTEBOOK' },
];

describe('ExperimentLoggedModelSourceBox', () => {
  const renderTestComponent = (tags = defaultTestTags) => {
    const mockLoggedModel: LoggedModelProto = {
      info: {
        tags,
      },
    };

    render(<ExperimentLoggedModelSourceBox loggedModel={mockLoggedModel} displayDetails />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <TestRouter routes={[testRoute(<>{children}</>)]} />
        </DesignSystemProvider>
      ),
    });
  };

  it('renders local source', async () => {
    renderTestComponent([
      ...defaultTestTags,
      { key: 'mlflow.source.type', value: 'LOCAL' },
      { key: 'mlflow.source.name', value: 'some-file.py' },
    ]);
    await waitFor(() => {
      expect(screen.getByText('some-file.py')).toBeInTheDocument();
    });
    expect(screen.getByText('branch-abc')).toBeInTheDocument();
    expect(screen.getByText('abc123d')).toBeInTheDocument();
  });

  it('renders a placeholder when run source is not defined', async () => {
    renderTestComponent([]);
    await waitFor(() => {
      expect(screen.getByText('—')).toBeInTheDocument();
    });
  });
});
