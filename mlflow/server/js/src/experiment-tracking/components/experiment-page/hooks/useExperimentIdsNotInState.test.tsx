import { mount } from 'enzyme';
import React from 'react';
import { Provider } from 'react-redux';
import { createStore } from 'redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../fixtures/experiment-runs.fixtures';
import { useExperimentIdsNotInState } from './useExperimentIdsNotInState';

describe('useExperiments', () => {
  const WrapComponent = (Component: React.ComponentType, store: any) => {
    return (
      <Provider store={createStore((s) => s as any, store)}>
        <Component />
      </Provider>
    );
  };
  it('Does not fetch single non-existing experiment from the store', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperimentIdsNotInState(['does-not-exist']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(1);
  });

  it('only selects multiple non-existing experiments', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperimentIdsNotInState(['nope', 'not-in-state']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(2);
  });

  it('does not fetch one existing experiments', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperimentIdsNotInState(['654321']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(0);
  });

  it('does not select multiple existing experiments', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperimentIdsNotInState(['123456789', '654321']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(0);
  });
});
