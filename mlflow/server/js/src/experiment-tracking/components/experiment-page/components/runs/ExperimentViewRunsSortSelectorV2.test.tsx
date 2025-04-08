import userEvent from '@testing-library/user-event';
import { render, screen } from '../../../../../common/utils/TestUtils.react18';
import { ExperimentViewRunsSortSelectorV2 } from './ExperimentViewRunsSortSelectorV2';
import { DesignSystemProvider } from '@databricks/design-system';
import { MemoryRouter, useSearchParams } from '../../../../../common/utils/RoutingUtils';
import { IntlProvider } from 'react-intl';

const metricKeys = ['metric_alpha', 'metric_beta'];
const paramKeys = ['param_1', 'param_2', 'param_3'];

jest.mock('../../../../../common/utils/RoutingUtils', () => {
  const params = new URLSearchParams();
  const setSearchParamsMock = jest.fn((setter: (newParams: URLSearchParams) => URLSearchParams) => setter(params));
  return {
    ...jest.requireActual<typeof import('../../../../../common/utils/RoutingUtils')>(
      '../../../../../common/utils/RoutingUtils',
    ),
    useSearchParams: () => {
      return [params, setSearchParamsMock];
    },
  };
});

describe('ExperimentViewRunsSortSelectorV2', () => {
  let mockLatestParams = new URLSearchParams();

  const getCurrentSearchParams = () => Object.fromEntries(mockLatestParams.entries());

  const renderComponent = () => {
    const Component = () => {
      const [latestParams] = useSearchParams();
      mockLatestParams = latestParams;

      return (
        <ExperimentViewRunsSortSelectorV2
          metricKeys={metricKeys}
          paramKeys={paramKeys}
          orderByKey=""
          orderByAsc={false}
        />
      );
    };
    render(<Component />, {
      wrapper: ({ children }) => (
        <MemoryRouter>
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        </MemoryRouter>
      ),
    });
  };

  beforeEach(() => {
    mockLatestParams = new URLSearchParams();
  });

  test('should change sort options', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'metric_alpha' }));

    expect(getCurrentSearchParams()).toEqual(expect.objectContaining({ orderByKey: 'metrics.`metric_alpha`' }));

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.click(screen.getByLabelText('Sort ascending'));

    expect(getCurrentSearchParams()).toEqual(
      expect.objectContaining({
        orderByAsc: 'true',
        orderByKey: 'metrics.`metric_alpha`',
      }),
    );

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.click(screen.getByLabelText('Sort descending'));

    expect(getCurrentSearchParams()).toEqual(
      expect.objectContaining({
        orderByAsc: 'false',
        orderByKey: 'metrics.`metric_alpha`',
      }),
    );

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'param_2' }));

    expect(getCurrentSearchParams()).toEqual(
      expect.objectContaining({
        orderByAsc: 'false',
        orderByKey: 'params.`param_2`',
      }),
    );
  });

  test('should search for parameter', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.type(screen.getByPlaceholderText('Search'), 'beta');
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'metric_beta' }));

    expect(getCurrentSearchParams()).toEqual(expect.objectContaining({ orderByKey: 'metrics.`metric_beta`' }));

    await userEvent.click(screen.getByRole('button', { name: /Sort/ }));
    await userEvent.type(screen.getByPlaceholderText('Search'), 'abc-xyz-foo-bar');
    expect(screen.getByText('No results')).toBeInTheDocument();
  });
});
