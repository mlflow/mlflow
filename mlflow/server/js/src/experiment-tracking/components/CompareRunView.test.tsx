import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../common/utils/TestUtils';
import { render, screen, waitFor } from '../../common/utils/TestUtils.react18';
import CompareRunView from './CompareRunView';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';

describe('CompareRunView', () => {
  const wrapper = ({ children }: { children?: React.ReactNode }) => (
    <IntlProvider locale="en">
      <TestRouter routes={[testRoute(<>{children}</>)]} />
    </IntlProvider>
  );
  test('Will display title for two runs', async () => {
    render(
      <MockedReduxStoreProvider
        state={
          {
            compareExperiments: {},
            comparedExperiments: {},
            entities: {
              experimentsById: { exp_1: {} },
              runInfosByUuid: { run_1: { runUuid: 'run_1' }, run_2: { runUuid: 'run_2' } },
              paramsByRunUuid: { run_1: {}, run_2: {} },
              latestMetricsByRunUuid: { run_1: {}, run_2: {} },
              tagsByRunUuid: { run_1: {}, run_2: {} },
            },
          } as any
        }
      >
        <CompareRunView experimentIds={['exp_1']} runUuids={['run_1', 'run_2']} />
      </MockedReduxStoreProvider>,
      {
        wrapper,
      },
    );

    await waitFor(() => {
      expect(screen.getByText(/Comparing 2 Runs/)).toBeInTheDocument();
    });
  });

  test('Will not crash when run infos are not present in the store', async () => {
    render(
      <MockedReduxStoreProvider
        state={
          {
            compareExperiments: {},
            comparedExperiments: {},
            entities: {
              experimentsById: { exp_1: {} },
              runInfosByUuid: {},
              paramsByRunUuid: {},
              latestMetricsByRunUuid: {},
            },
          } as any
        }
      >
        <CompareRunView experimentIds={['exp_1']} runUuids={['run_1', 'run_2']} />
      </MockedReduxStoreProvider>,
      {
        wrapper,
      },
    );

    await waitFor(() => {
      expect(screen.getByText(/Comparing 0 Runs/)).toBeInTheDocument();
    });
  });

  test('Displays visualizations section with parallel coordinates plot by default', async () => {
    render(
      <MockedReduxStoreProvider
        state={
          {
            compareExperiments: {},
            comparedExperiments: {},
            entities: {
              experimentsById: { exp_1: {} },
              runInfosByUuid: { run_1: { runUuid: 'run_1' }, run_2: { runUuid: 'run_2' } },
              paramsByRunUuid: { run_1: {}, run_2: {} },
              latestMetricsByRunUuid: { run_1: {}, run_2: {} },
              tagsByRunUuid: { run_1: {}, run_2: {} },
            },
          } as any
        }
      >
        <CompareRunView experimentIds={['exp_1']} runUuids={['run_1', 'run_2']} />
      </MockedReduxStoreProvider>,
      {
        wrapper,
      },
    );

    await waitFor(() => {
      expect(screen.getByText(/Visualizations/i)).toBeInTheDocument();
      expect(screen.getByText(/Parallel Coordinates Plot/i)).toBeInTheDocument();
    });
  });
});
