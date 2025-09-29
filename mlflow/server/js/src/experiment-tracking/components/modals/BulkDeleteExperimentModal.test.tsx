import { BulkDeleteExperimentModal } from './BulkDeleteExperimentModal';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../sdk/MlflowService';
import { deleteExperimentApi, getExperimentApi } from '../../actions';
import Utils from '../../../common/utils/Utils';

jest.mock('../../actions', () => ({
  getExperimentApi: jest.fn(() => ({ type: 'action', meta: {}, payload: Promise.resolve({}) })),
  deleteExperimentApi: jest.fn(() => ({ type: 'action', meta: {}, payload: Promise.resolve({}) })),
}));

describe('RenameExperimentModal', () => {
  let minimalProps: any;

  beforeEach(() => {
    jest.mocked(deleteExperimentApi).mockClear();
    jest.mocked(getExperimentApi).mockClear();
    jest.spyOn(MlflowService, 'getExperimentByName').mockImplementation(() => Promise.reject({} as any));
    jest.spyOn(Utils, 'logErrorAndNotifyUser');
    jest.clearAllMocks();
  });

  const renderTestComponent = () => {
    minimalProps = {
      isOpen: true,
      experiments: [{ experimentId: 0, name: '0' }],
      onClose: jest.fn(() => Promise.resolve({})),
      onExperimentsDeleted: jest.fn(),
    };

    render(<BulkDeleteExperimentModal {...minimalProps} />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <MockedReduxStoreProvider
            state={{
              entities: {},
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
    expect(screen.getByText(/Delete \d+ Experiment/)).toBeInTheDocument();
  });

  test('form submission should result in deleteExperimentApi calls', async () => {
    renderTestComponent();
    await userEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(deleteExperimentApi).toHaveBeenCalledTimes(1);
    });
  });

  test('if deleteExperimentApi fails, error is reported', async () => {
    const error = new Error('123');
    jest
      .mocked(deleteExperimentApi)
      .mockImplementation(() => ({ type: 'action', meta: {}, payload: Promise.reject(error) } as any));

    renderTestComponent();
    await userEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(Utils.logErrorAndNotifyUser).toHaveBeenLastCalledWith(error);
    });
  });
});
