import { IntlProvider } from 'react-intl';
import { render, screen, waitFor } from '@testing-library/react';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { shouldUseRegexpBasedAutoRunsSearchFilter } from '../../../../../common/utils/FeatureUtils';

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
  shouldUseRegexpBasedAutoRunsSearchFilter: jest.fn(),
}));

describe('RunsSearchAutoComplete', () => {
  const onClear = jest.fn();
  const onSearchFilterChange = jest.fn();

  const renderTestComponent = () => {
    render(
      <RunsSearchAutoComplete
        onClear={onClear}
        onSearchFilterChange={onSearchFilterChange}
        requestError={null}
        runsData={{
          datasetsList: [],
          experimentTags: {},
          metricKeyList: [],
          metricsList: [],
          modelVersionsByRunUuid: {},
          paramKeyList: [],
          paramsList: [],
          runInfos: [],
          runUuidsMatchingFilter: [],
          tagsList: [],
        }}
        searchFilter=""
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );
  };
  it('should render', async () => {
    // Enable automatic transformation from plain text to regexp-based RLIKE filter
    jest.mocked(shouldUseRegexpBasedAutoRunsSearchFilter).mockReturnValue(true);
    renderTestComponent();

    await userEvent.type(screen.getByRole('combobox'), 'foobar{Enter}');

    // Wait for the search filter to be updated
    await waitFor(() => {
      expect(onSearchFilterChange).toHaveBeenCalledWith('foobar');
    });

    // Expect relevant tooltip to be displayed
    expect(
      screen.getByLabelText(/The following query will be used: attributes\.run_name RLIKE 'foobar'/),
    ).toBeInTheDocument();
  });
});
