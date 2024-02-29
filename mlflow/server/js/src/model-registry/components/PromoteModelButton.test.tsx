import userEvent from '@testing-library/user-event';

import { DeepPartial } from 'redux';
import { MemoryRouter, useNavigate } from '../../common/utils/RoutingUtils';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { findAntdOption, act, screen, within, fastFillInput, renderWithIntl } from 'common/utils/TestUtils.react17';
import { PromoteModelButton } from './PromoteModelButton';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { Services as ModelRegistryService } from '../services';
import { ModelVersionStatus, Stages } from '../constants';
import { ReduxState } from '../../redux-types';
import { ModelRegistryRoutes } from '../routes';
import { merge } from 'lodash';

jest.mock('../../model-registry/services');
jest.mock('../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual('../../common/utils/RoutingUtils'),
  useNavigate: jest.fn(),
}));

// Mocks/blocks debounce() used in component in order to make sure it won't fire after test is finished
jest.useFakeTimers();

describe('PromoteModelButton', () => {
  const renderComponent = (partialState: DeepPartial<ReduxState> = {}) => {
    const mv = mockModelVersionDetailed('modelA', '1', Stages.PRODUCTION, ModelVersionStatus.READY);

    return renderWithIntl(
      <MockedReduxStoreProvider
        state={merge(
          {
            entities: {
              modelByName: {},
            },
          },
          partialState,
        )}
      >
        <MemoryRouter>
          <PromoteModelButton modelVersion={mv} />
        </MemoryRouter>
      </MockedReduxStoreProvider>,
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render the button', () => {
    renderComponent();
    const buttonElement = screen.getByRole('button', { name: 'Promote model' });
    expect(buttonElement).toBeInTheDocument();
  });

  it('prepopulates the search registry on render', () => {
    const searchRegistryMock = jest.fn();
    ModelRegistryService.searchRegisteredModels = searchRegistryMock;
    renderComponent();
    expect(searchRegistryMock).toHaveBeenCalledTimes(1);
  });

  it('should show the modal when the button is clicked', async () => {
    renderComponent();
    const buttonElement = screen.getByRole('button', { name: 'Promote model' });
    await act(async () => {
      userEvent.click(buttonElement);
    });
    const modalElement = screen.getByText(/Copy your MLflow models/);
    expect(modalElement).toBeInTheDocument();
    expect(modalElement).toBeVisible();
  });

  it('should hide the modal when the cancel button is clicked', async () => {
    renderComponent();
    const buttonElement = screen.getByRole('button', { name: 'Promote model' });
    await act(async () => {
      userEvent.click(buttonElement);
    });
    const cancelButtonElement = screen.getByRole('button', { name: 'Cancel' });
    await act(async () => {
      userEvent.click(cancelButtonElement);
    });
    const modalElement = screen.getByText(/Copy your MLflow models/);
    expect(modalElement).toBeInTheDocument();
    expect(modalElement).not.toBeVisible();
  });

  it('should invoke expected APIs for copy model version flow', async () => {
    // Mock this function minimally, only to be spied on
    jest.spyOn(ModelRegistryService, 'searchRegisteredModels').mockResolvedValue([]);

    // Mock this function to return a mock model version
    jest.spyOn(ModelRegistryService, 'createModelVersion').mockReturnValue(
      Promise.resolve({
        model_version: mockModelVersionDetailed('modelA', '2', Stages.PRODUCTION, ModelVersionStatus.READY),
      }),
    );

    // Mock the useNavigate hook
    const mockNavigate = jest.fn();
    (useNavigate as jest.Mock).mockReturnValue(mockNavigate);

    // Render the component with pre-populated redux state that already has a registered model entity
    renderComponent({
      entities: {
        modelByName: {
          modelA: mockRegisteredModelDetailed('modelA'),
        },
      },
    });

    // First, open the modal
    await act(async () => {
      userEvent.click(screen.getByRole('button', { name: 'Promote model' }));
    });

    // Then, fill in the select filter and click on the option
    await act(async () => {
      await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'modelA');
      userEvent.click(findAntdOption('modelA'));
    });

    // Assert two calls to search
    expect(ModelRegistryService.searchRegisteredModels).toHaveBeenCalledTimes(2);

    const explanationText = screen.queryByText(/will be copied to modelA/);
    expect(explanationText).toBeInTheDocument();

    // Click "copy" button
    await act(async () => {
      userEvent.click(screen.getByRole('button', { name: 'Promote' }));
    });

    // We should have a new model version created in the already existing model
    expect(ModelRegistryService.createModelVersion).toHaveBeenCalledTimes(1);
    expect(ModelRegistryService.createRegisteredModel).toHaveBeenCalledTimes(0);
    expect(mockNavigate).toHaveBeenCalledWith(ModelRegistryRoutes.getModelVersionPageRoute('modelA', '2'));
  });

  it('should invoke expected APIs for create registered model and copy MV flow', async () => {
    // Mock this function minimally, only to be spied on
    jest.spyOn(ModelRegistryService, 'searchRegisteredModels').mockResolvedValue([]);

    // Mock this function to return a mock model version
    jest.spyOn(ModelRegistryService, 'createModelVersion').mockReturnValue(
      Promise.resolve({
        model_version: mockModelVersionDetailed('modelB', '1', Stages.PRODUCTION, ModelVersionStatus.READY),
      }),
    );

    // Mock this function to return a mock registered model
    jest
      .spyOn(ModelRegistryService, 'createRegisteredModel')
      .mockReturnValue(Promise.resolve(mockRegisteredModelDetailed('modelB')));

    // Mock the useNavigate hook
    const mockNavigate = jest.fn();
    (useNavigate as jest.Mock).mockReturnValue(mockNavigate);

    renderComponent({
      entities: {
        modelByName: {
          modelA: mockRegisteredModelDetailed('modelA'),
        },
      },
    });

    // First, open the modal
    await act(async () => {
      userEvent.click(screen.getByRole('button', { name: 'Promote model' }));
    });

    // Select create new model option
    await act(async () => {
      userEvent.click(screen.getByRole('combobox'));
    });
    await act(async () => {
      userEvent.click(findAntdOption('Create New Model'));
    });

    // Assert two calls to search
    expect(ModelRegistryService.searchRegisteredModels).toHaveBeenCalledTimes(2);

    const explanationText = screen.queryByText(/will be copied/);
    expect(explanationText).not.toBeInTheDocument();

    // Fill in the model name
    await act(async () => {
      await fastFillInput(screen.getByPlaceholderText('Input a model name'), 'modelB');
    });

    // Click "copy" button
    await act(async () => {
      userEvent.click(screen.getByRole('button', { name: 'Promote' }));
    });

    // We should have a created a new registered model and a new model version
    expect(ModelRegistryService.createRegisteredModel).toHaveBeenCalledTimes(1);
    expect(ModelRegistryService.createModelVersion).toHaveBeenCalledTimes(1);
    expect(mockNavigate).toHaveBeenCalledWith(ModelRegistryRoutes.getModelVersionPageRoute('modelB', '1'));
  });
});
