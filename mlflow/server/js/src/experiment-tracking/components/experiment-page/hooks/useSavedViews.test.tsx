import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { renderHook, act } from '@testing-library/react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

import { useSavedViews } from './useSavedViews';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { encodeSavedViewEnvelope, getSavedViewTagKey } from '../utils/savedViewEnvelope';
import { ExperimentTag } from '../../../sdk/MlflowMessages';
import type { ExperimentEntity } from '../../../types';
import { DELETE_EXPERIMENT_TAG_API, GET_EXPERIMENT_API } from '../../../actions';

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useNavigate: jest.fn(),
}));

jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    setExperimentTag: jest.fn(() => Promise.resolve({})),
    deleteExperimentTag: jest.fn(() => Promise.resolve({})),
    getExperiment: jest.fn(() => Promise.resolve({ experiment: { experiment_id: 'exp-1', tags: [] } })),
  },
}));

const EXPERIMENT_ID = 'exp-1';

const makeExperiment = (allowedActions?: string[]): ExperimentEntity =>
  ({ experimentId: EXPERIMENT_ID, tags: [], allowedActions }) as unknown as ExperimentEntity;

// Build a redux state slice with the given saved-view envelopes seeded into
// experimentTagsByExperimentId (the live-updating source the hook reads from).
const makeState = (views: { id: string; name: string; createdAt: number; state?: string }[]) => {
  const tagObj: Record<string, unknown> = {};
  views.forEach(({ id, name, createdAt, state = 'deflate;xxx' }) => {
    const key = getSavedViewTagKey(id);
    tagObj[key] = (ExperimentTag as any).fromJs({ key, value: encodeSavedViewEnvelope(name, state, createdAt) });
  });
  return {
    entities: {
      experimentTagsByExperimentId: {
        [EXPERIMENT_ID]: tagObj,
      },
    },
  };
};

describe('useSavedViews', () => {
  const navigateMock = jest.fn();

  const renderUseSavedViews = (state: any, experiment: ExperimentEntity) => {
    const store = configureStore([thunk, promiseMiddleware()])(state);
    const result = renderHook(() => useSavedViews({ experiment }), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    return { ...result, store };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useNavigate).mockReturnValue(navigateMock as ReturnType<typeof useNavigate>);
  });

  it('lists saved views from redux tags sorted by createdAt descending', () => {
    const state = makeState([
      { id: 'a', name: 'Older', createdAt: 1000 },
      { id: 'b', name: 'Newer', createdAt: 3000 },
      { id: 'c', name: 'Middle', createdAt: 2000 },
    ]);
    const { result } = renderUseSavedViews(state, makeExperiment());

    expect(result.current.views.map((v) => v.name)).toEqual(['Newer', 'Middle', 'Older']);
    expect(result.current.views.map((v) => v.id)).toEqual(['b', 'c', 'a']);
  });

  it('dispatches deleteExperimentTagApi with the saved-view tag key when deleting a view', async () => {
    const state = makeState([{ id: 'todelete', name: 'Doomed', createdAt: 1000 }]);
    const { result, store } = renderUseSavedViews(state, makeExperiment());

    await act(async () => {
      await result.current.deleteView('todelete');
    });

    const actions = store.getActions();
    const delAction = actions.find((a: any) => a.type === `${DELETE_EXPERIMENT_TAG_API}_PENDING`);
    expect(delAction).toBeDefined();
    expect(delAction.meta.experimentId).toBe(EXPERIMENT_ID);
    expect(delAction.meta.key).toBe(getSavedViewTagKey('todelete'));
  });

  it('refetches the experiment then navigates to the runs tab with the view id in viewStateShareKey', async () => {
    const state = makeState([{ id: 'view-42', name: 'Open me', createdAt: 1000 }]);
    const { result, store } = renderUseSavedViews(state, makeExperiment());

    await act(async () => {
      await result.current.openView('view-42');
    });

    // The experiment is refetched first so the just-saved tag is present in `experiment.tags`
    // by the time the reader resolves the share key (the reader reads the experimentsById slice,
    // which only updates on GET_EXPERIMENT_API — not on the tag write).
    const actions = store.getActions();
    expect(actions.some((a: any) => a.type === `${GET_EXPERIMENT_API}_PENDING`)).toBe(true);

    expect(navigateMock).toHaveBeenCalledWith(expect.stringMatching(/viewStateShareKey=view-42/));
    expect(navigateMock).toHaveBeenCalledWith(expect.stringMatching(/\/experiments\/exp-1\/runs/));
  });

  it('reports canModify=true when the experiment has no allowedActions restriction', () => {
    const { result } = renderUseSavedViews(makeState([]), makeExperiment());
    expect(result.current.canModify).toBe(true);
  });

  it('reports canModify=false when the experiment lacks MODIFIY_PERMISSION', () => {
    const { result } = renderUseSavedViews(makeState([]), makeExperiment(['READ']));
    expect(result.current.canModify).toBe(false);
  });
});
