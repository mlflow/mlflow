import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import userEvent from '@testing-library/user-event';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';

import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';
import { useEvaluationArtifactWriteBack } from './useEvaluationArtifactWriteBack';
import {
  WRITE_BACK_EVALUATION_ARTIFACTS,
  discardPendingEvaluationData,
} from '../../../actions/PromptEngineeringActions';
import { uploadArtifactApi } from '../../../actions';
import { MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME } from '../../../constants';
import Utils from '../../../../common/utils/Utils';
import { fulfilled } from '../../../../common/utils/ActionUtils';

const mockState: EvaluationDataReduxState = {
  evaluationDraftInputValues: [],
  evaluationArtifactsByRunUuid: {},
  evaluationArtifactsLoadingByRunUuid: {},
  evaluationArtifactsErrorByRunUuid: {},
  evaluationPendingDataByRunUuid: {},
  evaluationPendingDataLoadingByRunUuid: {},
  evaluationArtifactsBeingUploaded: {},
};

jest.mock('../../../actions/PromptEngineeringActions', () => ({
  ...jest.requireActual<typeof import('../../../actions/PromptEngineeringActions')>(
    '../../../actions/PromptEngineeringActions',
  ),
  discardPendingEvaluationData: jest.fn().mockReturnValue({
    type: 'discardPendingEvaluationData',
    payload: Promise.resolve({}),
  }),
}));

jest.mock('../../../actions', () => ({
  getEvaluationTableArtifact: jest.fn().mockReturnValue({
    type: 'getEvaluationTableArtifact',
    payload: Promise.resolve({}),
  }),
  uploadArtifactApi: jest.fn().mockReturnValue({
    type: 'uploadArtifactApi',
    payload: Promise.resolve({}),
  }),
}));

const getPendingEntry = () => ({
  entryData: { question: 'new_question', answer: 'new_answer' },
  isPending: true,
  evaluationTime: 1,
});

describe('useEvaluationArtifactWriteBack + writeBackEvaluationArtifacts action', () => {
  let mockStore: any;
  const mountHook = (partialState: Partial<EvaluationDataReduxState> = {}) => {
    mockStore = configureStore([thunk, promiseMiddleware()])({
      evaluationData: { ...mockState, ...partialState },
    });
    const Component = () => {
      const { isSyncingArtifacts, EvaluationSyncStatusElement } = useEvaluationArtifactWriteBack();

      return (
        <div>
          {isSyncingArtifacts && <div data-testid="is-syncing" />}
          {EvaluationSyncStatusElement}
        </div>
      );
    };
    return renderWithIntl(
      <Provider store={mockStore}>
        <Component />
      </Provider>,
    );
  };
  it('properly displays entries to be evaluated', () => {
    const { container } = mountHook({
      evaluationPendingDataByRunUuid: {
        run_1: [getPendingEntry(), getPendingEntry()],
        run_2: [getPendingEntry()],
      },
    });

    expect(container).toHaveTextContent(/You have 3 unsaved evaluated values/);
  });

  beforeEach(() => {
    Utils.logErrorAndNotifyUser = jest.fn();
  });

  afterEach(() => {
    jest.mocked(Utils.logErrorAndNotifyUser).mockRestore();
  });

  it('properly synchronizes new entries', async () => {
    mountHook({
      evaluationArtifactsByRunUuid: {
        run_1: {
          [MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME]: {
            columns: ['question', 'answer'],
            entries: [],
            rawArtifactFile: {
              columns: ['question', 'answer'],
              data: [['existing_question', 'existing_answer']],
            },
            path: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
          },
        },
        run_2: {
          [MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME]: {
            columns: ['question', 'answer'],
            entries: [],
            rawArtifactFile: {
              columns: ['question', 'answer'],
              data: [],
            },
            path: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
          },
        },
      },
      evaluationPendingDataByRunUuid: {
        run_1: [getPendingEntry()],
        run_2: [getPendingEntry()],
      },
    });

    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(uploadArtifactApi).toHaveBeenCalledWith('run_1', MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME, {
      columns: ['question', 'answer'],
      data: [
        ['new_question', 'new_answer'],
        ['existing_question', 'existing_answer'],
      ],
    });

    expect(uploadArtifactApi).toHaveBeenCalledWith('run_2', MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME, {
      columns: ['question', 'answer'],
      data: [['new_question', 'new_answer']],
    });

    expect(mockStore.getActions()).toContainEqual(
      expect.objectContaining({
        meta: {
          artifactPath: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
          runUuidsToUpdate: ['run_1', 'run_2'],
        },
        payload: [
          {
            newEvaluationTable: {
              columns: ['question', 'answer'],
              // Two entries for run 1
              entries: [
                { answer: 'new_answer', question: 'new_question' },
                { answer: 'existing_answer', question: 'existing_question' },
              ],
              path: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
              rawArtifactFile: {
                columns: ['question', 'answer'],
                data: [
                  ['new_question', 'new_answer'],
                  ['existing_question', 'existing_answer'],
                ],
              },
            },
            runUuid: 'run_1',
          },
          {
            newEvaluationTable: {
              columns: ['question', 'answer'],
              // Only new entry for run 2
              entries: [{ answer: 'new_answer', question: 'new_question' }],
              path: MLFLOW_PROMPT_ENGINEERING_ARTIFACT_NAME,
              rawArtifactFile: {
                columns: ['question', 'answer'],
                data: [['new_question', 'new_answer']],
              },
            },
            runUuid: 'run_2',
          },
        ],
        type: fulfilled(WRITE_BACK_EVALUATION_ARTIFACTS),
      }),
    );
  });

  it('throws when the current artifact is not found in the store', async () => {
    mountHook({
      evaluationArtifactsByRunUuid: {},
      evaluationPendingDataByRunUuid: {
        run_1: [getPendingEntry()],
      },
    });

    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringMatching(/Cannot find existing prompt engineering artifact for run run_1/),
      }),
    );
  });

  it('discards the data when clicked', async () => {
    mountHook({
      evaluationArtifactsByRunUuid: {},
      evaluationPendingDataByRunUuid: {
        run_1: [getPendingEntry()],
      },
    });

    await userEvent.click(screen.getByRole('button', { name: 'Discard' }));

    expect(discardPendingEvaluationData).toHaveBeenCalledWith();
  });
});
