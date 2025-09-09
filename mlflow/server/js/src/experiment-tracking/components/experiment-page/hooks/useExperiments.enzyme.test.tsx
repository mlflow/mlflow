import { mount } from 'enzyme';
import React from 'react';
import { Provider } from 'react-redux';
import { createStore } from 'redux';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../fixtures/experiment-runs.fixtures';
import { useExperiments } from './useExperiments';

describe('useExperiments', () => {
  const WrapComponent = (Component: React.ComponentType<React.PropsWithChildren<unknown>>, store: any) => {
    return (
      <Provider store={createStore((s) => s as any, store)}>
        <Component />
      </Provider>
    );
  };
  it('fetches single experiment from the store properly', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperiments(['123456789']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(1);
  });

  it('fetches multiple experiments from the store properly', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperiments(['123456789', '654321']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(2);
  });

  it('does not select non-existing experiments', () => {
    let experiments: any;
    const Component = () => {
      experiments = useExperiments(['123', '321']);
      return null;
    };

    mount(WrapComponent(Component, EXPERIMENT_RUNS_MOCK_STORE));

    expect(experiments.length).toEqual(0);
  });
});
