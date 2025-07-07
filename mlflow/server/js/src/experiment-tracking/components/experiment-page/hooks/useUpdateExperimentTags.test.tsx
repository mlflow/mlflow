import { renderHook, act, waitFor, render, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { useUpdateExperimentTags } from './useUpdateExperimentTags';
import { ExperimentEntity } from '../../../types';
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

  // it('should call api services and success callback when edited and saved', async () => {
  //   const onSuccess = jest.fn();
  //   const { result, rerender } = renderTestHook(onSuccess);

  //   act(() => {
  //     result.current.showEditExperimentTagsModal(mockExperiment);
  //   });

  //   rerender();

  //   const { unmount } = render(<IntlProvider locale="en">{result.current.EditTagsModal}</IntlProvider>);

  //   await userEvent.click(within(screen.getByRole('status', { name: 'tag1' })).getByRole('button'));

  //   await userEvent.click(within(screen.getByRole('dialog')).getByRole('combobox'));
  //   await userEvent.paste('tag2', {
  //     clipboardData: { getData: jest.fn() },
  //   } as any);
  //   await userEvent.click(screen.getByText(/Add tag "tag2"/));

  //   const valueInput = screen.getByLabelText('Value');
  //   await userEvent.type(valueInput, 'value2');

  //   await userEvent.click(screen.getByLabelText('Add tag'));

  //   await userEvent.click(screen.getByRole('button', { name: 'Save tags' }));

  //   await waitFor(() => {
  //     expect(result.current.isLoading).toBe(true);
  //   });

  //   await waitFor(() => {
  //     expect(MlflowService.deleteExperimentTag).toHaveBeenCalledWith(mockExperiment.experimentId, 'tag1');
  //     expect(MlflowService.setExperimentTag).toHaveBeenCalledWith(mockExperiment.experimentId, 'tag2', 'value2');
  //     expect(onSuccess).toHaveBeenCalled();
  //   });

  //   await waitFor(() => {
  //     expect(result.current.isLoading).toBe(false);
  //   });

  //   expect(result.current.EditTagsModal.props.visible).toBeFalsy();
  //   unmount();
  // });
});
