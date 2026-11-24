import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import type { ReactNode } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import {
  renderHook,
  act,
  fireEvent,
  screen,
  renderWithDesignSystem,
} from '@mlflow/mlflow/src/common/utils/TestUtils.react18';

import { useSharedViewActions } from './useSharedViewActions';
import { loadExperimentViewState, saveExperimentViewState } from '../utils/persistSearchFacets';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import { createExperimentPageUIState } from '../models/ExperimentPageUIState';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import type { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';

jest.mock('../utils/persistSearchFacets', () => ({
  loadExperimentViewState: jest.fn(),
  saveExperimentViewState: jest.fn(),
}));

jest.mock('../../../../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/RoutingUtils')>(
    '../../../../common/utils/RoutingUtils',
  ),
  useSearchParams: jest.fn(),
}));

const PERSIST_KEY = JSON.stringify(['exp1']);

describe('useSharedViewActions', () => {
  const setSearchParamsMock = jest.fn();

  const wrapper = ({ children }: { children: ReactNode }) => (
    <IntlProvider locale="en">
      <DesignSystemProvider>{children}</DesignSystemProvider>
    </IntlProvider>
  );

  const renderActions = (overrides: Partial<Parameters<typeof useSharedViewActions>[0]> = {}) => {
    const setUIState = jest.fn();
    const exitSharedView = jest.fn();
    const props = {
      experimentIds: ['exp1'],
      searchFacets: {
        ...createExperimentPageSearchFacetsState(),
        orderByKey: 'metrics.m1',
      } as ExperimentQueryParamsSearchFacets,
      uiState: { ...createExperimentPageUIState(), viewMaximized: true },
      setUIState,
      exitSharedView,
      ...overrides,
    };
    const { result } = renderHook(() => useSharedViewActions(props), { wrapper });
    return { result, setUIState, exitSharedView, props };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(useSearchParams).mockReturnValue([new URLSearchParams(), setSearchParamsMock]);
    jest.mocked(loadExperimentViewState).mockReturnValue({});
  });

  it('override persists the current view, exits shared mode, strips the share key, and shows a toast', () => {
    const displaySpy = jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
    const { result, exitSharedView, props } = renderActions();

    act(() => result.current.handleOverrideSavedView());

    expect(loadExperimentViewState).toHaveBeenCalledWith(PERSIST_KEY);
    expect(saveExperimentViewState).toHaveBeenCalledWith(
      { ...props.searchFacets, ...props.uiState } as never,
      PERSIST_KEY,
    );
    expect(exitSharedView).toHaveBeenCalledTimes(1);

    // The URL rewrite must drop the share key and the preview flag while preserving live facets.
    const updater = setSearchParamsMock.mock.calls[0][0] as (p: URLSearchParams) => URLSearchParams;
    const params = updater(new URLSearchParams('viewStateShareKey=abc&isPreview=true&orderByKey=metrics.m1'));
    expect(params.has('viewStateShareKey')).toBe(false);
    expect(params.has('isPreview')).toBe(false);
    expect(params.get('orderByKey')).toBe('metrics.m1');

    expect(displaySpy).toHaveBeenCalledTimes(1);
  });

  it('override keeps the user’s personal (non-shareable) prefs instead of persisting the shared-view defaults', () => {
    // The applied shared view reset per-run/personal fields to defaults in `uiState`; Override must
    // restore them from the user's own saved view rather than wiping them.
    const previous = { runsPinned: ['my-pinned-run'], autoRefreshEnabled: true };
    jest.mocked(loadExperimentViewState).mockReturnValue(previous);
    jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
    const { result } = renderActions({
      uiState: { ...createExperimentPageUIState(), viewMaximized: true, runsPinned: [], autoRefreshEnabled: false },
    });

    act(() => result.current.handleOverrideSavedView());

    const saved = jest.mocked(saveExperimentViewState).mock.calls[0][0];
    // Shareable state comes from the current (shared) view...
    expect(saved.viewMaximized).toBe(true);
    // ...but non-shareable prefs are preserved from the user's saved view.
    expect(saved.runsPinned).toEqual(['my-pinned-run']);
    expect(saved.autoRefreshEnabled).toBe(true);
  });

  it('undo restores the previously saved view from the toast action', () => {
    const previous = { selectedColumns: ['metrics.previously_saved'] };
    jest.mocked(loadExperimentViewState).mockReturnValue(previous);
    let toastContent: ReactNode;
    jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation((content) => {
      toastContent = content as ReactNode;
    });
    const { result, setUIState } = renderActions();

    act(() => result.current.handleOverrideSavedView());

    // Render the captured toast and click Undo.
    renderWithDesignSystem(<>{toastContent}</>);
    fireEvent.click(screen.getByRole('button', { name: /undo/i }));

    expect(saveExperimentViewState).toHaveBeenLastCalledWith(previous as never, PERSIST_KEY);
    expect(setUIState).toHaveBeenCalledWith(
      expect.objectContaining({ selectedColumns: ['metrics.previously_saved'] }) as never,
    );
  });

  it('discard restores the saved view, rewrites the URL, and exits shared mode', () => {
    const saved = { selectedColumns: ['metrics.saved'], orderByKey: 'metrics.saved' };
    jest.mocked(loadExperimentViewState).mockReturnValue(saved);
    const { result, setUIState, exitSharedView } = renderActions();

    act(() => result.current.handleDiscardSharedView());

    expect(loadExperimentViewState).toHaveBeenCalledWith(PERSIST_KEY);
    expect(setUIState).toHaveBeenCalledWith(expect.objectContaining({ selectedColumns: ['metrics.saved'] }) as never);
    expect(exitSharedView).toHaveBeenCalledTimes(1);

    const updater = setSearchParamsMock.mock.calls[0][0] as (p: URLSearchParams) => URLSearchParams;
    const params = updater(new URLSearchParams('viewStateShareKey=abc'));
    expect(params.has('viewStateShareKey')).toBe(false);
    expect(params.get('orderByKey')).toBe('metrics.saved');
  });
});
