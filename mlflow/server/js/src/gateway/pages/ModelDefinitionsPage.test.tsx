import { describe, jest, beforeEach, test, expect } from '@jest/globals';
import { renderWithDesignSystem, screen } from '../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';

// Mock the child components to avoid needing to mock all nested hooks
jest.mock('../components/model-definitions/ModelDefinitionsList', () => ({
  ModelDefinitionsList: () => <div data-testid="model-definitions-list">Model Definitions List</div>,
}));
jest.mock('../components/model-definitions/CreateModelDefinitionModal', () => ({
  CreateModelDefinitionModal: () => null,
}));
jest.mock('../components/model-definitions/DeleteModelDefinitionModal', () => ({
  DeleteModelDefinitionModal: () => null,
}));
jest.mock('../hooks/useModelDefinitionsQuery', () => ({
  useModelDefinitionsQuery: () => ({
    data: [],
    isLoading: false,
    error: undefined,
    refetch: jest.fn(),
  }),
}));
jest.mock('../hooks/useEndpointsQuery', () => ({
  useEndpointsQuery: () => ({
    data: [],
    isLoading: false,
    error: undefined,
    refetch: jest.fn(),
  }),
}));

// Import after mocks are set up
import ModelDefinitionsPage from './ModelDefinitionsPage';

describe('ModelDefinitionsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders page with header and create button', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <ModelDefinitionsPage />
      </MemoryRouter>,
    );

    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('Create Model')).toBeInTheDocument();
  });

  test('renders model definitions list component', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <ModelDefinitionsPage />
      </MemoryRouter>,
    );

    expect(screen.getByTestId('model-definitions-list')).toBeInTheDocument();
  });
});
