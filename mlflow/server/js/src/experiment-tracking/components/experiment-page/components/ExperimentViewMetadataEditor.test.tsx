import { describe, beforeEach, expect, jest, test } from '@jest/globals';
import { renderWithDesignSystem, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';
import type { ExperimentEntity } from '../../../types';
import { ExperimentViewMetadataEditor } from './ExperimentViewMetadataEditor';
import { getExperimentApi, setExperimentTagApi, updateExperimentApi } from '../../../actions';
import { MlflowService } from '../../../sdk/MlflowService';
import { ErrorCodes } from '../../../../common/constants';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { encodeTraceArchivalRetentionTag } from '../../../../common/utils/traceArchival';

const mockDispatch = jest.fn((..._args: any[]) => Promise.resolve());
const mockUseSelector = jest.fn();
const mockInvalidateExperimentList = jest.fn();

jest.mock('react-redux', () => ({
  useDispatch: () => mockDispatch,
  useSelector: (selector: any) => mockUseSelector(selector),
}));

jest.mock('../../../actions', () => ({
  getExperimentApi: jest.fn(),
  setExperimentTagApi: jest.fn(),
  updateExperimentApi: jest.fn(),
}));

jest.mock('../hooks/useExperimentListQuery', () => ({
  useInvalidateExperimentList: () => mockInvalidateExperimentList,
}));

jest.mock('../../../../common/components/EditableNote', () => ({
  ThemeAwareReactMde: ({ value, onChange, childProps }: any) => (
    <textarea {...childProps?.textArea} value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}));

const defaultExperiment: ExperimentEntity = {
  experimentId: '123',
  name: 'test/experiment/name',
  artifactLocation: 'file:/tmp/mlruns',
  lifecycleStage: 'active',
  allowedActions: ['RENAME', 'MODIFIY_PERMISSION'],
  creationTime: 0,
  lastUpdateTime: 0,
  tags: [{ key: 'mlflow.trace.archivalRetention', value: encodeTraceArchivalRetentionTag('30d') }],
};

describe('ExperimentViewMetadataEditor', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseSelector.mockImplementation((selector: any) =>
      selector({
        entities: {
          experimentTagsByExperimentId: {},
          experimentsById: {
            '123': defaultExperiment,
            '456': { ...defaultExperiment, experimentId: '456', name: 'existing-experiment' },
          },
        },
      }),
    );
    jest
      .spyOn(MlflowService, 'getExperimentByName')
      .mockImplementation(() =>
        Promise.reject(new ErrorWrapper({ error_code: ErrorCodes.RESOURCE_DOES_NOT_EXIST, message: 'not found' }, 404)),
      );
    jest
      .mocked(setExperimentTagApi)
      .mockImplementation(
        (experimentId, tagName, tagValue) =>
          ({ type: 'MOCK_TAG', payload: { experimentId, tagName, tagValue } }) as any,
      );
    jest
      .mocked(updateExperimentApi)
      .mockImplementation(
        (experimentId, newExperimentName) =>
          ({ type: 'MOCK_RENAME', payload: { experimentId, newExperimentName } }) as any,
      );
    jest
      .mocked(getExperimentApi)
      .mockImplementation((experimentId) => ({ type: 'MOCK_GET_EXPERIMENT', payload: { experimentId } }) as any);
    jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({} as any);
    jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue({} as any);
  });

  test('renders edit experiment modal and saves renamed experiment metadata', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.getByText('Edit experiment')).toBeInTheDocument();
    expect(screen.getByDisplayValue('test/experiment/name')).toBeInTheDocument();
    expect(screen.getByLabelText('Trace archival retention')).toHaveValue('30');

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(updateExperimentApi).toHaveBeenCalledWith('123', 'renamed-experiment');
    });
    expect(setExperimentTagApi).not.toHaveBeenCalled();
    expect(getExperimentApi).toHaveBeenCalledWith('123');
    expect(mockInvalidateExperimentList).toHaveBeenCalled();
  });

  test('associates the trace archival retention label with the amount input', () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.getByLabelText('Trace archival retention')).toHaveAttribute(
      'id',
      'mlflow.experiment.edit.trace-archival-retention-amount',
    );
  });

  test('does not save the description when the rename fails', async () => {
    const dispatchedActions: string[] = [];
    mockDispatch.mockImplementation((action: any) => {
      dispatchedActions.push(action.type);
      if (action.type === 'MOCK_RENAME') {
        return Promise.reject(new Error('Rename failed'));
      }
      return Promise.resolve();
    });

    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.clear(screen.getByRole('textbox', { name: 'Description' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Description' }), 'Updated description');
    await userEvent.click(screen.getByText('Save'));

    expect(await screen.findByText('Rename failed')).toBeInTheDocument();
    expect(setExperimentTagApi).not.toHaveBeenCalled();
    expect(getExperimentApi).not.toHaveBeenCalled();
    expect(dispatchedActions).toEqual(['MOCK_RENAME']);
    expect(mockInvalidateExperimentList).not.toHaveBeenCalled();
  });

  test('treats the post-rename refresh as best effort', async () => {
    const dispatchedActions: string[] = [];
    const setEditing = jest.fn();
    mockDispatch.mockImplementation((action: any) => {
      dispatchedActions.push(action.type);
      if (action.type === 'MOCK_GET_EXPERIMENT') {
        return Promise.reject(new Error('Refresh failed'));
      }
      return Promise.resolve();
    });

    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={setEditing}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(setEditing).toHaveBeenCalledWith(false);
    });
    expect(dispatchedActions).toEqual(['MOCK_RENAME', 'MOCK_GET_EXPERIMENT']);
    expect(screen.queryByText('Refresh failed')).not.toBeInTheDocument();
    expect(mockInvalidateExperimentList).toHaveBeenCalled();
  });

  test('shows validation error when experiment name is empty', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.click(screen.getByText('Save'));

    expect(await screen.findByText('Please input a new name for the experiment.')).toBeInTheDocument();
    expect(updateExperimentApi).not.toHaveBeenCalled();
  });

  test('only shows the rename field when rename is the only allowed edit action', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={{ ...defaultExperiment, allowedActions: ['RENAME'] }}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.getByDisplayValue('test/experiment/name')).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Description' })).not.toBeInTheDocument();

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(updateExperimentApi).toHaveBeenCalledWith('123', 'renamed-experiment');
    });
    expect(setExperimentTagApi).not.toHaveBeenCalled();
  });

  test('only shows metadata fields when metadata modification is the only allowed edit action', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={{ ...defaultExperiment, allowedActions: ['MODIFIY_PERMISSION'] }}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.queryByPlaceholderText('Enter experiment name')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Description' })).toBeInTheDocument();

    await userEvent.clear(screen.getByRole('textbox', { name: 'Description' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Description' }), 'Updated description');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(setExperimentTagApi).toHaveBeenCalledWith('123', 'mlflow.note.content', 'Updated description');
    });
    expect(updateExperimentApi).not.toHaveBeenCalled();
  });

  test('saves description changes without triggering rename', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByRole('textbox', { name: 'Description' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Description' }), 'Updated description');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(setExperimentTagApi).toHaveBeenCalledWith('123', 'mlflow.note.content', 'Updated description');
    });
    expect(updateExperimentApi).not.toHaveBeenCalled();
    expect(getExperimentApi).not.toHaveBeenCalled();
    expect(mockInvalidateExperimentList).not.toHaveBeenCalled();
  });

  test('does not update description when nothing changed', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(setExperimentTagApi).not.toHaveBeenCalled();
    });
    expect(updateExperimentApi).not.toHaveBeenCalled();
    expect(getExperimentApi).not.toHaveBeenCalled();
    expect(mockInvalidateExperimentList).not.toHaveBeenCalled();
  });
  test('preserves an intentionally cleared description instead of falling back to the default', async () => {
    mockUseSelector.mockImplementation((selector: any) =>
      selector({
        entities: {
          experimentTagsByExperimentId: {
            '123': {
              clearedNote: { key: 'mlflow.note.content', value: '' },
            },
          },
          experimentsById: {
            '123': defaultExperiment,
            '456': { ...defaultExperiment, experimentId: '456', name: 'existing-experiment' },
          },
        },
      }),
    );
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );
    expect(screen.getByRole('textbox', { name: 'Description' })).toHaveValue('');

    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(setExperimentTagApi).not.toHaveBeenCalled();
    });
  });

  test('saves updated trace archival retention', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '14');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(MlflowService.setExperimentTag).toHaveBeenCalledWith({
        experiment_id: '123',
        key: 'mlflow.trace.archivalRetention',
        value: encodeTraceArchivalRetentionTag('14d'),
      });
    });
    expect(getExperimentApi).toHaveBeenCalledWith('123');
    expect(MlflowService.deleteExperimentTag).not.toHaveBeenCalled();
  });

  test('clears trace archival retention override', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(MlflowService.deleteExperimentTag).toHaveBeenCalledWith({
        experiment_id: '123',
        key: 'mlflow.trace.archivalRetention',
      });
    });
  });

  test('shows validation error for invalid trace archival retention', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '0');
    await userEvent.click(screen.getByText('Save'));

    expect(
      await screen.findByText(
        "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
      ),
    ).toBeInTheDocument();
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
  });

  test('preserves invalid retention characters so validation feedback is visible', async () => {
    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '1a');

    expect(screen.getByLabelText('Trace archival retention')).toHaveValue('1a');
  });
  test('only shows the inline edit button when metadata modification is allowed', () => {
    const { rerender } = renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={{ ...defaultExperiment, allowedActions: ['RENAME'] }}
        editing={false}
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.queryByTestId('experiment-metadata-editor-edit-button')).not.toBeInTheDocument();

    rerender(
      <ExperimentViewMetadataEditor
        experiment={{ ...defaultExperiment, allowedActions: ['MODIFIY_PERMISSION'] }}
        editing={false}
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    expect(screen.getByTestId('experiment-metadata-editor-edit-button')).toBeInTheDocument();
  });

  test('refreshes once after saving rename and retention together', async () => {
    const callOrder: string[] = [];
    jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async () => {
      callOrder.push('retention');
      return {} as any;
    });
    mockDispatch.mockImplementation((action: any) => {
      if (action.type === 'MOCK_RENAME') {
        callOrder.push('rename');
        return Promise.resolve();
      }
      if (action.type === 'MOCK_GET_EXPERIMENT') {
        callOrder.push('refresh');
        return Promise.resolve();
      }
      return Promise.resolve();
    });

    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '14');
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(callOrder).toEqual(['rename', 'retention', 'refresh']);
    });
    expect(getExperimentApi).toHaveBeenCalledTimes(1);
    expect(mockInvalidateExperimentList).toHaveBeenCalled();
  });

  test('does not apply retention or description changes when rename fails after editing multiple fields', async () => {
    const callOrder: string[] = [];
    jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async () => {
      callOrder.push('retention');
      return {} as any;
    });
    mockDispatch.mockImplementation((action: any) => {
      if (action.type === 'MOCK_RENAME') {
        callOrder.push('rename');
        return Promise.reject(new Error('Rename failed'));
      }
      if (action.type === 'MOCK_GET_EXPERIMENT') {
        callOrder.push('refresh');
        return Promise.resolve();
      }
      return Promise.resolve();
    });

    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.clear(screen.getByRole('textbox', { name: 'Description' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Description' }), 'Updated description');
    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '14');
    await userEvent.click(screen.getByText('Save'));

    expect(await screen.findByText('Rename failed')).toBeInTheDocument();
    expect(setExperimentTagApi).not.toHaveBeenCalled();
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
    expect(MlflowService.deleteExperimentTag).not.toHaveBeenCalled();
    expect(getExperimentApi).not.toHaveBeenCalled();
    expect(callOrder).toEqual(['rename']);
    expect(mockInvalidateExperimentList).not.toHaveBeenCalled();
  });

  test('shows a partial success error when rename succeeds, description fails, and retention is not updated', async () => {
    mockDispatch.mockImplementation((action: any) => {
      if (action.type === 'MOCK_RENAME') {
        return Promise.resolve();
      }
      if (action.type === 'MOCK_TAG') {
        return Promise.reject(new Error('Tag update failed'));
      }
      if (action.type === 'MOCK_GET_EXPERIMENT') {
        return Promise.resolve();
      }
      return Promise.resolve();
    });

    renderWithDesignSystem(
      <ExperimentViewMetadataEditor
        experiment={defaultExperiment}
        editing
        setEditing={jest.fn()}
        defaultValue="Existing description"
      />,
    );

    await userEvent.clear(screen.getByDisplayValue('test/experiment/name'));
    await userEvent.type(screen.getByPlaceholderText('Enter experiment name'), 'renamed-experiment');
    await userEvent.clear(screen.getByRole('textbox', { name: 'Description' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Description' }), 'Updated description');
    await userEvent.clear(screen.getByLabelText('Trace archival retention'));
    await userEvent.type(screen.getByLabelText('Trace archival retention'), '14');
    await userEvent.click(screen.getByText('Save'));

    expect(
      await screen.findByText('Experiment renamed. Description update failed. Trace archival retention not updated.'),
    ).toBeInTheDocument();
    expect(MlflowService.setExperimentTag).not.toHaveBeenCalled();
    expect(MlflowService.deleteExperimentTag).not.toHaveBeenCalled();
    expect(getExperimentApi).toHaveBeenCalledWith('123');
    expect(mockInvalidateExperimentList).toHaveBeenCalled();
  });
});
