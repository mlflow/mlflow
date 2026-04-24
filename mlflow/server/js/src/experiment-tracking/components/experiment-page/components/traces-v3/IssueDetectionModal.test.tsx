import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen, waitFor } from '../../../../../common/utils/TestUtils.react18';
import { IssueDetectionModal } from './IssueDetectionModal';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import { useInvokeIssueDetection } from './hooks/useInvokeIssueDetection';
import { useLocation, useNavigate } from '../../../../../common/utils/RoutingUtils';

jest.mock('../../../../../gateway/hooks/useCreateSecret');
jest.mock('./hooks/useInvokeIssueDetection');
jest.mock('../../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/RoutingUtils')>(
    '../../../../../common/utils/RoutingUtils',
  ),
  useNavigate: jest.fn(),
  useLocation: jest.fn(),
}));
let mockModelSelectionValues: {
  mode: 'direct' | 'endpoint';
  provider: string;
  model: string;
  apiKeyConfig: {
    mode: 'new' | 'existing';
    existingSecretId: string;
    newSecret: {
      name: string;
      authMode: string;
      secretFields: Record<string, string>;
      configFields: Record<string, string>;
    };
  };
  saveKey: boolean;
} = {
  mode: 'direct',
  provider: '',
  model: '',
  apiKeyConfig: {
    mode: 'new',
    existingSecretId: '',
    newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
  },
  saveKey: true,
};
let mockModelSelectionValid = false;

jest.mock('./GenAIModelSelection', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  return {
    GenAIModelSelection: React.forwardRef(function GenAIModelSelection(
      {
        onValidityChange,
      }: {
        onValidityChange: (isValid: boolean) => void;
      },
      ref: any,
    ) {
      React.useImperativeHandle(ref, () => ({
        getValues: () => mockModelSelectionValues,
        isValid: mockModelSelectionValid,
        reset: () => {
          mockModelSelectionValues = {
            mode: 'direct',
            provider: '',
            model: '',
            apiKeyConfig: {
              mode: 'new',
              existingSecretId: '',
              newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
            },
            saveKey: true,
          };
          mockModelSelectionValid = false;
        },
      }));

      return (
        <div data-testid="model-selection">
          <button
            data-testid="set-valid-existing-key"
            onClick={() => {
              mockModelSelectionValues = {
                mode: 'direct',
                provider: 'openai',
                model: 'gpt-5-mini',
                apiKeyConfig: {
                  mode: 'existing',
                  existingSecretId: 'secret-123',
                  newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
                },
                saveKey: false,
              };
              mockModelSelectionValid = true;
              onValidityChange(true);
            }}
          >
            Use existing key
          </button>
          <button
            data-testid="set-valid-new-key"
            onClick={() => {
              mockModelSelectionValues = {
                mode: 'direct',
                provider: 'openai',
                model: 'gpt-5-mini',
                apiKeyConfig: {
                  mode: 'new',
                  existingSecretId: '',
                  newSecret: { name: 'my-key', authMode: '', secretFields: { api_key: 'sk-123' }, configFields: {} },
                },
                saveKey: true,
              };
              mockModelSelectionValid = true;
              onValidityChange(true);
            }}
          >
            Use new key
          </button>
          <button
            data-testid="set-invalid"
            onClick={() => {
              mockModelSelectionValid = false;
              onValidityChange(false);
            }}
          >
            Set invalid
          </button>
        </div>
      );
    }),
  };
});

jest.mock('../../../SelectTracesModal', () => ({
  SelectTracesModal: ({
    onClose,
    onSuccess,
    defaultGroupBySession,
  }: {
    onClose: () => void;
    onSuccess: (traceIds: string[]) => void;
    defaultGroupBySession?: boolean;
  }) => (
    <div data-testid="select-traces-modal">
      <div data-testid="default-group-by-session">{String(defaultGroupBySession)}</div>
      <button data-testid="select-traces-cancel" onClick={onClose}>
        Cancel
      </button>
      <button data-testid="select-traces-confirm" onClick={() => onSuccess(['trace-1', 'trace-2'])}>
        Select
      </button>
    </div>
  ),
}));

describe('IssueDetectionModal', () => {
  const defaultProps = {
    onClose: jest.fn(),
    experimentId: 'exp-123',
  };

  let mockCreateSecret: jest.Mock;
  let mockResetCreateSecret: jest.Mock;
  let mockInvokeIssueDetection: jest.Mock;
  let mockResetIssueDetection: jest.Mock;
  let mockNavigate: jest.Mock;

  beforeEach(() => {
    jest.clearAllMocks();
    mockNavigate = jest.fn();
    jest.mocked(useNavigate).mockReturnValue(mockNavigate);
    jest.mocked(useLocation).mockReturnValue({ search: '', pathname: '/', hash: '', state: null, key: 'default' });
    // Reset mock values
    mockModelSelectionValues = {
      mode: 'direct',
      provider: '',
      model: '',
      apiKeyConfig: {
        mode: 'new',
        existingSecretId: '',
        newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
      },
      saveKey: true,
    };
    mockModelSelectionValid = false;

    mockCreateSecret = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { secret: { secret_id: string } }) => void })?.onSuccess?.({
        secret: { secret_id: 'new-secret-123' },
      });
    });
    mockResetCreateSecret = jest.fn();
    mockInvokeIssueDetection = jest.fn((_request, options) => {
      (options as { onSuccess?: (response: { job_id: string; run_id: string }) => void })?.onSuccess?.({
        job_id: 'job-123',
        run_id: 'run-456',
      });
    });
    mockResetIssueDetection = jest.fn();
    jest.mocked(useCreateSecret).mockReturnValue({
      mutate: mockCreateSecret,
      isLoading: false,
      error: null,
      reset: mockResetCreateSecret,
    } as any);
    jest.mocked(useInvokeIssueDetection).mockReturnValue({
      mutate: mockInvokeIssueDetection,
      isLoading: false,
      error: null,
      reset: mockResetIssueDetection,
    } as any);
  });

  // Helper to navigate to step 2 (provider/model configuration)
  const navigateToStep2 = async () => {
    const nextButton = screen.getByText('Next').closest('button')!;
    await userEvent.click(nextButton);
  };

  test('renders modal with step 1 (category selection)', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
    expect(screen.getByText('Select Categories')).toBeInTheDocument();
  });

  test('renders description text', () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    expect(
      screen.getByText('Use AI to automatically analyze your traces and identify potential issues'),
    ).toBeInTheDocument();
  });

  test('renders model selection in step 2', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();

    expect(screen.getByTestId('model-selection')).toBeInTheDocument();
  });

  test('submit button is disabled when form is invalid', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    // Form starts invalid
    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).toBeDisabled();

    // Set form to valid state (model valid + traces already selected)
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));
    expect(submitButton).not.toBeDisabled();

    // Set form back to invalid state
    await userEvent.click(screen.getByTestId('set-invalid'));
    expect(submitButton).toBeDisabled();
  });

  test('submit button is enabled when form is valid with existing key', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('submit button is enabled when form is valid with new key', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} />);

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button');
    expect(submitButton).not.toBeDisabled();
  });

  test('calls onClose when cancel button is clicked', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} onClose={onClose} />);

    const cancelButton = screen.getByText('Cancel').closest('button')!;
    await userEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  test('calls onClose when submit is clicked with existing key', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });

  test('shows trace count when initial traces are provided', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1', 'trace-2', 'trace-3']} />,
    );

    await navigateToStep2();

    expect(screen.getByText('3 traces selected')).toBeInTheDocument();
  });

  test('opens select traces modal when button is clicked', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('select-traces'));

    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();
  });

  test('updates trace count after selecting traces', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('select-traces'));
    await userEvent.click(screen.getByTestId('select-traces-confirm'));

    expect(screen.getByText('2 traces selected')).toBeInTheDocument();
  });

  test('closes select traces modal when cancel is clicked', async () => {
    renderWithDesignSystem(<IssueDetectionModal {...defaultProps} />);

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('select-traces'));
    expect(screen.getByTestId('select-traces-modal')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('select-traces-cancel'));
    expect(screen.queryByTestId('select-traces-modal')).not.toBeInTheDocument();
  });

  test('saves secret when form is submitted with new key', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-new-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockCreateSecret).toHaveBeenCalledWith(
        {
          secret_name: 'my-key',
          secret_value: { api_key: 'sk-123' },
          provider: 'openai',
          auth_config: undefined,
        },
        expect.any(Object),
      );
    });
  });

  test('does not save secret when using existing key', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
    expect(mockCreateSecret).not.toHaveBeenCalled();
  });

  test('navigates to run details page when form is submitted', async () => {
    const onClose = jest.fn();

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
      expect(mockNavigate).toHaveBeenCalledWith({
        pathname: '/experiments/exp-123/evaluation-runs/run-456',
        search: undefined,
      });
    });
  });

  test('preserves only time range query params when navigating to run details', async () => {
    const onClose = jest.fn();
    jest.mocked(useLocation).mockReturnValue({
      search: '?startTimeLabel=LAST_7_DAYS&someOtherParam=foo',
      pathname: '/',
      hash: '',
      state: null,
      key: 'default',
    });

    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} onClose={onClose} initialSelectedTraceIds={['trace-1']} />,
    );

    await navigateToStep2();
    await userEvent.click(screen.getByTestId('set-valid-existing-key'));

    const submitButton = screen.getByText('Run Analysis').closest('button')!;
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith({
        pathname: '/experiments/exp-123/evaluation-runs/run-456',
        search: '?startTimeLabel=LAST_7_DAYS',
      });
    });
  });

  test('passes defaultGroupBySession prop to SelectTracesModal when set to true', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} defaultGroupBySession />,
    );

    await navigateToStep2();

    // Open the select traces modal
    const selectTracesButton = screen.getByTestId('select-traces');
    await userEvent.click(selectTracesButton);

    // Verify the SelectTracesModal receives defaultGroupBySession=true
    expect(screen.getByTestId('default-group-by-session')).toHaveTextContent('true');
  });

  test('passes defaultGroupBySession prop to SelectTracesModal when set to false', async () => {
    renderWithDesignSystem(
      <IssueDetectionModal {...defaultProps} initialSelectedTraceIds={['trace-1']} defaultGroupBySession={false} />,
    );

    await navigateToStep2();

    // Open the select traces modal
    const selectTracesButton = screen.getByTestId('select-traces');
    await userEvent.click(selectTracesButton);

    // Verify the SelectTracesModal receives defaultGroupBySession=false
    expect(screen.getByTestId('default-group-by-session')).toHaveTextContent('false');
  });
});
