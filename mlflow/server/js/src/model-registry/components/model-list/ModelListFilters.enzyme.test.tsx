import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import type { ModelListFiltersProps } from './ModelListFilters';
import { ModelListFilters } from './ModelListFilters';

describe('ModelListFilters', () => {
  const minimalProps: ModelListFiltersProps = {
    isFiltered: false,
    onSearchFilterChange: () => {},
    searchFilter: '',
  };

  const createComponentWrapper = (moreProps: Partial<ModelListFiltersProps> = {}) => {
    return mountWithIntl(<ModelListFilters {...minimalProps} {...moreProps} />);
  };

  it('mounts the component and checks if input is rendered', () => {
    const wrapper = createComponentWrapper({});

    expect(wrapper.find('[data-testid="model-search-input"]')).toBeTruthy();
  });
  it('mounts the component and checks if search input works', () => {
    const onSearchFilterChange = jest.fn();
    const wrapper = createComponentWrapper({
      onSearchFilterChange,
    });

    wrapper.find('input[data-testid="model-search-input"]').simulate('change', { target: { value: 'searched model' } });

    wrapper.find('input[data-testid="model-search-input"]').simulate('submit');

    expect(onSearchFilterChange).toHaveBeenCalledWith('searched model');
  });

  it('resets the search filter', () => {
    const onSearchFilterChange = jest.fn();
    const wrapper = createComponentWrapper({
      onSearchFilterChange,
      searchFilter: 'some search filter',
      isFiltered: true,
    });

    wrapper.find('[data-testid="models-list-filters-reset"]').hostNodes().simulate('click');
    expect(onSearchFilterChange).toHaveBeenCalledWith('');
  });
});
