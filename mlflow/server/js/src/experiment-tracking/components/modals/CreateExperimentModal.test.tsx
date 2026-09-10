import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { act, render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import userEvent from '@testing-library/user-event';
import { CreateExperimentModal } from './CreateExperimentModal';
import { createExperimentApi } from '../../actions';
import { MlflowService } from '../../sdk/MlflowService';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { ErrorCodes } from '../../../common/constants';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import Utils from '../../../common/utils/Utils';

const mockNavigate = jest.fn();
jest.mock('../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/RoutingUtils')>('../../../common/utils/RoutingUtils'),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../actions', () => ({
  createExperimentApi: jest.fn(() => ({ type: 'action', meta: {}, payload: Promise.resolve({}) })),
}));

describe('CreateExperimentModal', () => {
  const notFoundError = new ErrorWrapper({ error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST, message: 'not found' }, 404);

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(MlflowService, 'getExperimentByName').mockImplementation(() => Promise.reject(notFoundError));
  });

  const renderTestComponent = (props: Partial<React.ComponentProps<typeof CreateExperimentModal>> = {}) => {
    const defaultProps = {
      isOpen: true,
      onClose: jest.fn(),
      onExperimentCreated: jest.fn(),
    };

    return render(<CreateExperimentModal {...defaultProps} {...props} />, {
      wrapper: ({ children }) => (
        <DesignSystemProvider>
          <IntlProvider locale="en">
            <MockedReduxStoreProvider
              state={{
                entities: {
                  experimentsById: {},
                },
              }}
            >
              {children}
            </MockedReduxStoreProvider>
          </IntlProvider>
        </DesignSystemProvider>
      ),
    });
  };

  test('renders create experiment modal with name and artifact fields', () => {
    renderTestComponent();
    expect(screen.getByText('Create Experiment')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /experiment name/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /artifact location/i })).toBeInTheDocument();
  });

  test('Create button is disabled when experiment name is empty', () => {
    renderTestComponent();
    const createButton = screen.getByRole('button', { name: /create/i });
    expect(createButton).toBeDisabled();
  });

  test('Create button is enabled when experiment name is entered', async () => {
    renderTestComponent();
    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, 'my-experiment');

    const createButton = screen.getByRole('button', { name: /create/i });
    expect(createButton).toBeEnabled();
  });

  test('Create button is disabled and required-field message shows when experiment name is cleared', async () => {
    renderTestComponent();
    const input = screen.getByRole('textbox', { name: /experiment name/i });

    await userEvent.type(input, 'my-experiment');
    expect(screen.getByRole('button', { name: /create/i })).toBeEnabled();

    await userEvent.clear(input);
    expect(screen.getByRole('button', { name: /create/i })).toBeDisabled();
    expect(screen.getByText(/please input a new name/i)).toBeInTheDocument();
  });

  test('Create button is disabled for whitespace-only experiment name', async () => {
    renderTestComponent();
    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, '   ');

    const createButton = screen.getByRole('button', { name: /create/i });
    expect(createButton).toBeDisabled();
  });

  test('redirects user to newly-created experiment page on successful creation', async () => {
    const fakeExperimentId = 'fakeExpId';
    jest.mocked(createExperimentApi).mockImplementation(
      () =>
        ({
          type: 'action',
          meta: {},
          payload: Promise.resolve({ experiment_id: fakeExperimentId }),
        }) as any,
    );

    const onExperimentCreated = jest.fn();
    renderTestComponent({ onExperimentCreated });

    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, 'myNewExp');

    const createButton = screen.getByRole('button', { name: /create/i });
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(createExperimentApi).toHaveBeenCalledWith('myNewExp', undefined);
      expect(onExperimentCreated).toHaveBeenCalled();
      expect(mockNavigate).toHaveBeenCalledWith(createMLflowRoutePath('/experiments/fakeExpId'));
    });
  });

  test('passes artifact location to createExperimentApi when provided', async () => {
    jest.mocked(createExperimentApi).mockImplementation(
      () =>
        ({
          type: 'action',
          meta: {},
          payload: Promise.resolve({ experiment_id: 'newId' }),
        }) as any,
    );

    renderTestComponent();

    await userEvent.type(screen.getByRole('textbox', { name: /experiment name/i }), 'myExp');
    await userEvent.type(screen.getByRole('textbox', { name: /artifact location/i }), 's3://my-bucket');

    await userEvent.click(screen.getByRole('button', { name: /create/i }));

    await waitFor(() => {
      // @ts-expect-error -- createExperimentApi has loosely typed params
      expect(createExperimentApi).toHaveBeenCalledWith('myExp', 's3://my-bucket');
    });
  });

  test('calls onClose when modal X button is clicked', async () => {
    const onClose = jest.fn();
    renderTestComponent({ onClose });

    await userEvent.click(screen.getByRole('button', { name: /close/i }));

    expect(onClose).toHaveBeenCalled();
  });

  test('blocks submit and shows inline error for a known duplicate name', async () => {
    const renderWithExistingExperiment = (props: Partial<React.ComponentProps<typeof CreateExperimentModal>> = {}) => {
      const defaultProps = {
        isOpen: true,
        onClose: jest.fn(),
        onExperimentCreated: jest.fn(),
      };

      return render(<CreateExperimentModal {...defaultProps} {...props} />, {
        wrapper: ({ children }) => (
          <DesignSystemProvider>
            <IntlProvider locale="en">
              <MockedReduxStoreProvider
                state={{
                  entities: {
                    experimentsById: {
                      '1': {
                        experimentId: '1',
                        name: 'existing-experiment',
                        artifactLocation: '',
                        creationTime: 0,
                        lastUpdateTime: 0,
                        lifecycleStage: 'active',
                        tags: [],
                      },
                    },
                  },
                }}
              >
                {children}
              </MockedReduxStoreProvider>
            </IntlProvider>
          </DesignSystemProvider>
        ),
      });
    };

    renderWithExistingExperiment();

    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, 'existing-experiment');

    const createButton = screen.getByRole('button', { name: /create/i });
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(screen.getByText(/already exists/i)).toBeInTheDocument();
    });
    expect(createExperimentApi).not.toHaveBeenCalled();
  });

  test('blocks submit and shows error for deleted experiment found via API', async () => {
    jest
      .spyOn(MlflowService, 'getExperimentByName')
      .mockImplementationOnce(() => Promise.resolve({ experiment: { lifecycleStage: 'deleted' } }) as any);

    renderTestComponent();

    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, 'deleted-experiment');

    await userEvent.click(screen.getByRole('button', { name: /create/i }));

    await waitFor(() => {
      expect(screen.getByText(/deleted state/i)).toBeInTheDocument();
    });
    expect(createExperimentApi).not.toHaveBeenCalled();
  });

  describe('Create button disabled after async validator sets error', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });
    afterEach(() => {
      jest.useRealTimers();
    });

    test('Create button is disabled after debounced validator finds an active duplicate', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      jest
        .spyOn(MlflowService, 'getExperimentByName')
        .mockImplementation(() => Promise.resolve({ experiment: { lifecycleStage: 'active' } }) as any);

      renderTestComponent();
      const input = screen.getByRole('textbox', { name: /experiment name/i });
      await user.type(input, 'taken-name');

      await act(async () => {
        jest.advanceTimersByTime(500);
      });

      await waitFor(() => {
        expect(screen.getByText(/already exists/i)).toBeInTheDocument();
      });
      expect(screen.getByRole('button', { name: /create/i })).toBeDisabled();
      expect(createExperimentApi).not.toHaveBeenCalled();
    });

    test('Create button is disabled after debounced validator finds a deleted experiment', async () => {
      const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
      jest
        .spyOn(MlflowService, 'getExperimentByName')
        .mockImplementation(() => Promise.resolve({ experiment: { lifecycleStage: 'deleted' } }) as any);

      renderTestComponent();
      const input = screen.getByRole('textbox', { name: /experiment name/i });
      await user.type(input, 'deleted-name');

      await act(async () => {
        jest.advanceTimersByTime(500);
      });

      await waitFor(() => {
        expect(screen.getByText(/deleted state/i)).toBeInTheDocument();
      });
      expect(screen.getByRole('button', { name: /create/i })).toBeDisabled();
      expect(createExperimentApi).not.toHaveBeenCalled();
    });
  });

  test('does not redirect or call onExperimentCreated when API request fails, and shows toast', async () => {
    const logSpy = jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
    jest.mocked(createExperimentApi).mockImplementation(
      () =>
        ({
          type: 'action',
          meta: {},
          payload: Promise.reject(new Error('CreateExperiment failed!')),
        }) as any,
    );

    const onExperimentCreated = jest.fn();
    renderTestComponent({ onExperimentCreated });

    const input = screen.getByRole('textbox', { name: /experiment name/i });
    await userEvent.type(input, 'myNewExp');

    const createButton = screen.getByRole('button', { name: /create/i });
    await userEvent.click(createButton);

    await waitFor(() => {
      expect(mockNavigate).not.toHaveBeenCalled();
      expect(onExperimentCreated).not.toHaveBeenCalled();
      expect(logSpy).toHaveBeenCalled();
    });

    logSpy.mockRestore();
  });
});
