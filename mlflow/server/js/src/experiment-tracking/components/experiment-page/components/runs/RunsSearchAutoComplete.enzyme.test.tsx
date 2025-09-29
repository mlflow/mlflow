import React, { useState } from 'react';
import { DesignSystemProvider, Input } from '@databricks/design-system';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../fixtures/experiment-runs.fixtures';
import { experimentRunsSelector } from '../../utils/experimentRuns.selector';
import { RunsSearchAutoComplete } from './RunsSearchAutoComplete';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';

const MOCK_EXPERIMENT = EXPERIMENT_RUNS_MOCK_STORE.entities.experimentsById['123456789'];

const MOCK_RUNS_DATA = experimentRunsSelector(EXPERIMENT_RUNS_MOCK_STORE, {
  experiments: [MOCK_EXPERIMENT],
});

const onClearMock = jest.fn();

const doStatefulMock = (additionalProps?: any) => {
  const mockUpdateSearchFacets = jest.fn();
  let currentState: any;

  const getCurrentState = () => currentState;

  const Component = () => {
    const [searchFacetsState, setSearchFacetsState] = useState(() => createExperimentPageSearchFacetsState());

    currentState = searchFacetsState;

    const updateSearchFacets = (updatedFilterState: Partial<ExperimentPageSearchFacetsState>) => {
      mockUpdateSearchFacets(updatedFilterState);
      if (typeof updatedFilterState === 'function') {
        setSearchFacetsState(updatedFilterState);
      } else {
        setSearchFacetsState((s: any) => ({ ...s, ...updatedFilterState }));
      }
    };

    const props = {
      runsData: MOCK_RUNS_DATA,
      searchFilter: searchFacetsState.searchFilter,
      onSearchFilterChange: (newSearchFilter: string) => {
        updateSearchFacets({ searchFilter: newSearchFilter });
      },
      onClear: onClearMock,
    } as any;
    return (
      <DesignSystemProvider>
        <RunsSearchAutoComplete {...{ ...props, ...additionalProps }} />
      </DesignSystemProvider>
    );
  };
  return {
    wrapper: mountWithIntl(<Component />),
    mockUpdateSearchFacets,
    getCurrentState,
  };
};

describe('AutoComplete', () => {
  test('Dialog has correct base options', () => {
    const { wrapper } = doStatefulMock();
    const groups = (wrapper.find('AutoComplete').props() as any).options;
    wrapper.update();
    expect(groups.length).toBe(4);
  });

  test('Dialog has correct options when starting to type metric name', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    // Should only have metric option group
    const props = wrapper.find('AutoComplete').props() as any;
    const groups = props.options;
    expect(props.open).toBe(true);
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Metrics');
  });

  test('Dialog opens when typing full metric name', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    const props = wrapper.find('AutoComplete').props() as any;
    const groups = props.options;
    expect(props.open).toBe(true);
    expect(groups.length).toBe(1);
    expect(groups[0].options[0].value).toBe('metrics.met1');
  });

  test('Dialog closes when typing entity name and comparator', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1<' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    const props = wrapper.find('AutoComplete').props() as any;
    expect(props.open).toBe(false);
  });

  test('Dialog is not open with a full search clause', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    const props = wrapper.find('AutoComplete').props() as any;
    expect(props.open).toBe(false);
  });

  test('Dialog is not open with `and` and no space', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3 and' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    const props = wrapper.find('AutoComplete').props() as any;
    expect(props.open).toBe(false);
  });

  test('Dialog opens with `and` and a space', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3 and' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3 and ' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    const props = wrapper.find('AutoComplete').props() as any;
    const groups = props.options;
    expect(props.open).toBe(true);
    expect(groups.length).toBe(4); // all groups should be visible here
  });

  test('Options are targeted at entity last edited', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3 and params.p1 < 2' } });
    searchBox.simulate('change', { target: { value: 'metrics.met1 < 3 and params.p < 2' } });

    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();

    let groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Parameters');
    expect(groups[0].options.length).toBe(3); // all params should be shown here
    expect(groups[0].options[0].value).toBe('params.p1');
    expect(groups[0].options[1].value).toBe('params.p2');

    searchBox.simulate('change', { target: { value: 'metrics.met < 3 and params.p < 2' } });
    wrapper.update();
    groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Metrics');
    expect(groups[0].options.length).toBe(3);
    expect(groups[0].options[0].value).toBe('metrics.met1');
    expect(groups[0].options[2].value).toBe('metrics.met3');
  });

  test('Dialog stays open when typing an entity name with it', () => {
    const { wrapper } = doStatefulMock();
    const searchBox = wrapper.find("input[data-testid='search-box']");

    searchBox.simulate('change', { target: { value: 'tags.' } });
    (wrapper.find(Input).prop('onClick') as any)();
    wrapper.update();
    let groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Tags');
    expect(groups[0].options.length).toBe(4);

    searchBox.simulate('change', { target: { value: 'tags.`' } });
    wrapper.update();
    groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Tags');
    expect(groups[0].options.length).toBe(1);
    expect(groups[0].options[0].value).toBe('tags.`tag with a space`');

    searchBox.simulate('change', { target: { value: 'tags.` ' } });
    wrapper.update();
    groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Tags');
    expect(groups[0].options.length).toBe(1);
    expect(groups[0].options[0].value).toBe('tags.`tag with a space`');

    searchBox.simulate('change', { target: { value: 'tags.`tag w' } });
    wrapper.update();
    groups = (wrapper.find('AutoComplete').props() as any).options;
    expect(groups.length).toBe(1);
    expect(groups[0].label).toBe('Tags');
    expect(groups[0].options.length).toBe(1);
    expect(groups[0].options[0].value).toBe('tags.`tag with a space`');
  });
});

describe('Input', () => {
  test('should update search facets model when searching by query', () => {
    const props = {
      runsData: MOCK_RUNS_DATA,
      onSearchFilterChange: jest.fn(),
      requestError: null,
    };

    const { wrapper } = doStatefulMock(props);

    const searchInput = wrapper.find("input[data-testid='search-box']");
    searchInput.simulate('change', { target: { value: 'test-query' } });
    searchInput.simulate('keydown', { key: 'Enter' });
    // First keydown dismisses autocomplete, second will search
    searchInput.simulate('keydown', { key: 'Enter' });

    expect(props.onSearchFilterChange).toHaveBeenCalledWith('test-query');
  });

  test('should update search facets model and properly clear filters afterwards', () => {
    const { wrapper, getCurrentState } = doStatefulMock();

    wrapper.find("input[data-testid='search-box']").simulate('change', { target: { value: 'test-query' } });
    wrapper.find("input[data-testid='search-box']").simulate('keydown', { key: 'Enter' });
    // First keydown dismisses autocomplete, second will search
    wrapper.find("input[data-testid='search-box']").simulate('keydown', { key: 'Enter' });

    expect(getCurrentState()).toMatchObject(expect.objectContaining({ searchFilter: 'test-query' }));

    expect(onClearMock).not.toHaveBeenCalled();

    wrapper.find("input[data-testid='search-box']").simulate('click');
    wrapper.find("button[data-testid='clear-button']").simulate('click');

    expect(onClearMock).toHaveBeenCalled();
  });

  test('should pop up tooltip when search returns error', () => {
    const searchFacetsState = createExperimentPageSearchFacetsState();

    const props = {
      runsData: MOCK_RUNS_DATA,
      updateSearchFacets: jest.fn(),
      searchFacetsState,
      requestError: new ErrorWrapper('some error'),
    };
    const { wrapper } = doStatefulMock(props);

    const toolTip = wrapper.find('.search-input-tooltip-content');

    expect(toolTip).toBeTruthy();
  });
});
