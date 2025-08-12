import { render, screen } from '@testing-library/react';
import { ExperimentLoggedModelTableNameCell } from './ExperimentLoggedModelTableNameCell';
import type { LoggedModelProto } from '../../types';
import { DesignSystemProvider } from '@databricks/design-system';
import { useExperimentLoggedModelRegisteredVersions } from './hooks/useExperimentLoggedModelRegisteredVersions';
import { BrowserRouter } from '../../../common/utils/RoutingUtils';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';

// Mock the hooks and safex
jest.mock('./hooks/useExperimentLoggedModelRegisteredVersions');
const mockUseExperimentLoggedModelRegisteredVersions = jest.mocked(useExperimentLoggedModelRegisteredVersions);
describe('ExperimentLoggedModelTableNameCell', () => {
  const createMockLoggedModel = (
    experimentId = 'exp-123',
    modelId = 'model-456',
    name = 'test-logged-model',
  ): LoggedModelProto => ({
    info: {
      experiment_id: experimentId,
      model_id: modelId,
      name,
    },
  });

  const renderComponent = (loggedModel: LoggedModelProto) => {
    render(<ExperimentLoggedModelTableNameCell data={loggedModel} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <BrowserRouter>{children}</BrowserRouter>
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Default hook mock - access granted and no models
    mockUseExperimentLoggedModelRegisteredVersions.mockReturnValue({
      modelVersions: [],
      isLoading: false,
    });
  });
  describe('when feature flag is disabled', () => {
    it('should display original logged model name and link to logged model details', () => {
      const loggedModel = createMockLoggedModel();
      renderComponent(loggedModel);

      expect(screen.getByText('test-logged-model')).toBeInTheDocument();
      expect(screen.queryByText('v1')).not.toBeInTheDocument();
    });

    it('should render without link when experiment_id or model_id is missing', () => {
      const loggedModel: LoggedModelProto = {
        info: {
          experiment_id: undefined,
          model_id: undefined,
          name: 'test-logged-model',
        },
      };
      renderComponent(loggedModel);

      expect(screen.getByText('test-logged-model')).toBeInTheDocument();
      expect(screen.queryByRole('link')).not.toBeInTheDocument();
    });

    it('should not display tooltip when feature flag is disabled', async () => {
      const userEvent_ = userEvent.setup();
      const loggedModel = createMockLoggedModel('exp-123', 'model-456', 'test-logged-model');
      renderComponent(loggedModel);

      const modelLink = screen.getByRole('link');
      await userEvent_.hover(modelLink);

      // Tooltip should not appear when feature flag is disabled
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
      expect(screen.queryByText(/Original logged model:/)).not.toBeInTheDocument();
    });
  });
});
