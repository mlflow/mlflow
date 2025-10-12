import { applyMiddleware, combineReducers, compose, createStore } from 'redux';
import { colorByRunUuid } from '../../../reducers/RunColorReducer';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider, useDispatch } from 'react-redux';
import { RunColorPill } from '../components/RunColorPill';
import {
  useGetExperimentRunColor,
  useInitializeExperimentRunColors,
  useSaveExperimentRunColor,
} from './useExperimentRunColor';
import { MlflowService } from '../../../sdk/MlflowService';
import type { ThunkDispatch } from '../../../../redux-types';
import { searchRunsApi } from '../../../actions';
import userEventGlobal from '@testing-library/user-event';

import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { useEffect } from 'react';

const userEvent = userEventGlobal.setup({
  advanceTimers: jest.advanceTimersByTime,
});

const run_1_uuid = 'run-1';
const run_2_uuid = 'run-2';
const group_1_uuid = 'group-1';

jest.useFakeTimers();

describe('useExperimentRunColor - integration test', () => {
  const renderTestComponent = (initialColors: Record<string, string> = {}) => {
    const rootReducer = combineReducers({
      entities: combineReducers({
        colorByRunUuid,
      }),
    });
    const store = createStore(
      rootReducer,
      {
        entities: {
          colorByRunUuid: { ...initialColors },
        },
      },
      compose(applyMiddleware(thunk, promiseMiddleware())),
    );

    const TestComponent = () => {
      const getColor = useGetExperimentRunColor();
      const saveColor = useSaveExperimentRunColor();
      useInitializeExperimentRunColors();
      const dispatch = useDispatch<ThunkDispatch>();

      const fetchRuns = async () => {
        dispatch(searchRunsApi({ experiment_id: '0' }));
      };

      return (
        <>
          <button onClick={fetchRuns}>Download runs data</button>
          <div data-testid="run-1">
            <RunColorPill
              color={getColor(run_1_uuid)}
              onChangeColor={(colorValue) =>
                saveColor({
                  runUuid: run_1_uuid,
                  colorValue,
                })
              }
            />
          </div>
          <div data-testid="run-2">
            <RunColorPill
              color={getColor(run_2_uuid)}
              onChangeColor={(colorValue) =>
                saveColor({
                  runUuid: run_2_uuid,
                  colorValue,
                })
              }
            />
          </div>
          <div data-testid="group-1">
            <RunColorPill
              color={getColor(group_1_uuid)}
              onChangeColor={(colorValue) =>
                saveColor({
                  groupUuid: group_1_uuid,
                  colorValue,
                })
              }
            />
          </div>
        </>
      );
    };

    const { rerender } = render(<TestComponent />, {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });

    return () => rerender(<TestComponent />);
  };

  const findColorPicker = (which: 'run-1' | 'run-2' | 'group-1') =>
    within(screen.getByTestId(which)).getByLabelText(/./);

  const selectColorUsingPicker = (color: string, whichPicker: 'run-1' | 'run-2' | 'group-1') =>
    fireEvent.input(findColorPicker(whichPicker), { target: { value: color } });

  beforeEach(() => {
    jest.spyOn(MlflowService, 'setTag').mockImplementation(() => Promise.resolve());
  });

  test('should change colors for both runs and group', async () => {
    renderTestComponent();

    // Set color for the first picker
    selectColorUsingPicker('#ff0000', run_1_uuid);

    act(() => {
      // Progress debounced function
      jest.advanceTimersToNextTimer();
    });

    // Assert that the color was saved in the API
    expect(MlflowService.setTag).toHaveBeenLastCalledWith({
      key: 'mlflow.runColor',
      run_uuid: run_1_uuid,
      value: '#ff0000',
    });
    // Assert that the color was changed
    expect(findColorPicker(run_1_uuid)).toHaveValue('#ff0000');

    // Set color for the second picker
    selectColorUsingPicker('#00ff00', run_2_uuid);

    act(() => {
      // Progress debounced function
      jest.advanceTimersToNextTimer();
    });

    // Assert that the color was saved in the API
    expect(MlflowService.setTag).toHaveBeenLastCalledWith({
      key: 'mlflow.runColor',
      run_uuid: run_2_uuid,
      value: '#00ff00',
    });
    // Assert that the color was changed
    expect(findColorPicker(run_2_uuid)).toHaveValue('#00ff00');

    // Set color for the group's picker
    selectColorUsingPicker('#0000ff', group_1_uuid);

    act(() => {
      // Progress debounced function
      jest.advanceTimersToNextTimer();
    });

    // Assert that the color was changed
    expect(findColorPicker(group_1_uuid)).toHaveValue('#0000ff');
  });

  test('should use initialized values for runs and intercept API responses with updated colors', async () => {
    // Mock the API response for the first run
    jest.spyOn(MlflowService, 'searchRuns').mockImplementation(() =>
      Promise.resolve({
        runs: [
          {
            info: { runUuid: run_1_uuid },
            data: { tags: [{ key: 'mlflow.runColor', value: '#ff00ff' }] },
          },
        ],
      } as any),
    );

    // Initialize the component with some data already in the store
    const rerender = renderTestComponent({
      [run_1_uuid]: '#ffff00',
    });

    // Assert that the color was displayed properly
    expect(findColorPicker(run_1_uuid)).toHaveValue('#ffff00');

    // Trigger the API response
    await userEvent.click(screen.getByText('Download runs data'));

    // Assert that the color was updated
    await waitFor(async () => {
      expect(findColorPicker(run_1_uuid)).toHaveValue('#ff00ff');
    });

    // Re-render the component to check if the color was persisted in the redux store
    rerender();

    // Assert that the color was persisted
    expect(findColorPicker(run_1_uuid)).toHaveValue('#ff00ff');
  });

  test('should use initialized values for groups and persist it in the local storage', async () => {
    // Initialize the component with some data already in the local storage
    window.localStorage.setItem('experimentRunColors', JSON.stringify({ [group_1_uuid]: '#ffff00' }));

    renderTestComponent();

    // Assert that the color was displayed properly
    expect(findColorPicker(group_1_uuid)).toHaveValue('#ffff00');

    // Set color for the group's picker
    selectColorUsingPicker('#ff00ff', group_1_uuid);

    act(() => {
      // Progress debounced function
      jest.advanceTimersToNextTimer();
    });

    // Clean up everything completely and re-render the component
    cleanup();
    renderTestComponent();

    // Assert that the color was persisted in the local storage
    expect(findColorPicker(group_1_uuid)).toHaveValue('#ff00ff');
  });
});
