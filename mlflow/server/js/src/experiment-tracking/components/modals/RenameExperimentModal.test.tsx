import { RenameExperimentModal } from './RenameExperimentModal';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../sdk/MlflowService';
import { getExperimentApi, updateExperimentApi } from '../../actions';
import Utils from '../../../common/utils/Utils';

jest.mock('../../actions', () => ({
  getExperimentApi: jest.fn(() => ({ type: 'action', meta: {}, payload: Promise.resolve({}) })),
  updateExperimentApi: jest.fn(() => ({ type: 'action', meta: {}, payload: Promise.resolve({}) })),
}));

describe('RenameExperimentModal', () => {
  let wrapper: any;
  let minimalProps: any;
  let mockUpdateExperimentApi: any;
  let mockGetExperimentApi: any;

  beforeEach(() => {
    jest.mocked(updateExperimentApi).mockClear();
    jest.mocked(getExperimentApi).mockClear();
    jest.spyOn(MlflowService, 'getExperimentByName').mockImplementation(() => Promise.reject({} as any));
    jest.spyOn(Utils, 'logErrorAndNotifyUser');
    jest.clearAllMocks();
  });

  const renderTestComponent = () => {
    minimalProps = {
      isOpen: true,
      experimentId: '123',
      experimentName: 'testName',
      experimentNames: ['arrayName1', 'arrayName2'],
      onClose: jest.fn(() => Promise.resolve({})),
      updateExperimentApi: mockUpdateExperimentApi,
      getExperimentApi: mockGetExperimentApi,
    };

    render(<RenameExperimentModal {...minimalProps} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <MockedReduxStoreProvider
            state={{
              entities: { experimentsById: {} },
            }}
          >
            {children}
          </MockedReduxStoreProvider>
        </IntlProvider>
      ),
    });
  };

  test('should render with minimal props without exploding', async () => {
    renderTestComponent();
    expect(screen.getByText('Rename Experiment')).toBeInTheDocument();
  });

  test('form submission should result in updateExperimentApi and getExperimentApi calls', async () => {
    renderTestComponent();
    await userEvent.clear(screen.getByLabelText('New experiment name'));
    await userEvent.type(screen.getByLabelText('New experiment name'), 'renamed');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(updateExperimentApi).toHaveBeenCalledTimes(1);
      expect(updateExperimentApi).toHaveBeenCalledWith('123', 'renamed');
      expect(getExperimentApi).toHaveBeenCalledTimes(1);
    });
  });

  test('if updateExperimentApi fails, error is reported', async () => {
    const error = new Error('123');
    jest
      .mocked(updateExperimentApi)
      .mockImplementation(() => ({ type: 'action', meta: {}, payload: Promise.reject(error) } as any));

    renderTestComponent();
    await userEvent.clear(screen.getByLabelText('New experiment name'));
    await userEvent.type(screen.getByLabelText('New experiment name'), 'renamed');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(Utils.logErrorAndNotifyUser).toHaveBeenLastCalledWith(error);
    });
  });

  test('if getExperimentApi fails, error is reported', async () => {
    const error = new Error('123');
    jest
      .mocked(getExperimentApi)
      .mockImplementation(() => ({ type: 'action', meta: {}, payload: Promise.reject(error) } as any));

    renderTestComponent();
    await userEvent.clear(screen.getByLabelText('New experiment name'));
    await userEvent.type(screen.getByLabelText('New experiment name'), 'renamed');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(Utils.logErrorAndNotifyUser).toHaveBeenLastCalledWith(error);
    });
  });
});
