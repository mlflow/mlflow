import {
  renderHook,
  act,
  waitFor,
  render,
  screen,
  within,
  fastFillInput,
  renderWithIntl,
} from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { useUpdateExperimentTags } from './useUpdateExperimentTags';
import type { ExperimentEntity } from '../../../types';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event';

jest.mock('../../../../common/utils/LocalStorageUtils');

const mockExperiment = {
  experiment_id: '12345',
  name: 'test-experiment',
  tags: [{ key: 'tag1', value: 'value1' }],
} as unknown as ExperimentEntity;

describe('useUpdateExperimentTags', () => {
  beforeEach(() => {
    jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue({});
  });

  function renderTestComponent(onSuccess: () => void) {
    function TestComponent() {
      const { showEditExperimentTagsModal, EditTagsModal } = useUpdateExperimentTags({ onSuccess });
      return (
        <>
          <button onClick={() => showEditExperimentTagsModal(mockExperiment)}>trigger button</button>
          {EditTagsModal}
        </>
      );
    }
    renderWithIntl(
      <QueryClientProvider client={new QueryClient()}>
        <TestComponent />
      </QueryClientProvider>,
    );
  }

  const renderTestHook = (onSuccess: () => void) =>
    renderHook(() => useUpdateExperimentTags({ onSuccess }), {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
        </IntlProvider>
      ),
    });

  it('should show nothing initially', () => {
    const onSuccess = jest.fn();
    const { result } = renderTestHook(onSuccess);

    expect(result.current.EditTagsModal.props.visible).toBeFalsy();
    expect(result.current.isLoading).toBe(false);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('should show edit modal if called with experiment', async () => {
    const onSuccess = jest.fn();
    const { result } = renderTestHook(onSuccess);

    act(() => {
      result.current.showEditExperimentTagsModal(mockExperiment);
    });

    await waitFor(() => {
      expect(result.current.EditTagsModal).not.toBeNull();
    });

    expect(result.current.EditTagsModal.props.visible).toBeTruthy();
    expect(result.current.isLoading).toBe(false);
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('should call api services and success callback when edited and saved', async () => {
    const onSuccess = jest.fn();
    renderTestComponent(onSuccess);

    await userEvent.click(screen.getByRole('button', { name: 'trigger button' }));

    expect(screen.getByRole('dialog', { name: /Add\/Edit tags/ })).toBeInTheDocument();
    await userEvent.click(within(screen.getByRole('status', { name: 'tag1' })).getByRole('button'));

    await fastFillInput(within(screen.getByRole('dialog')).getByRole('combobox'), 'tag2');

    await userEvent.click(screen.getByText(/Add tag "tag2"/));
    await fastFillInput(screen.getByLabelText('Value'), 'value2');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

    await waitFor(() => {
      expect(MlflowService.deleteExperimentTag).toHaveBeenCalledWith({
        experiment_id: mockExperiment.experimentId,
        key: 'tag1',
      });
      expect(MlflowService.setExperimentTag).toHaveBeenCalledWith({
        experiment_id: mockExperiment.experimentId,
        key: 'tag2',
        value: 'value2',
      });
      expect(onSuccess).toHaveBeenCalled();
    });
  });
});
