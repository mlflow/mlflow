import React, { Component } from 'react';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage } from 'react-intl';
// eslint-disable-next-line no-unused-vars
import { Link, withRouter } from 'react-router-dom';
import { Alert, Badge, Descriptions, Icon, Menu, Popover, Select, Tooltip } from 'antd';

import './ExperimentView.css';
import { getExperimentTags, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { setExperimentTagApi } from '../actions';
import Routes from '../routes';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import KeyFilter from '../utils/KeyFilter';
import { ExperimentRunsTableMultiColumnView2 } from './ExperimentRunsTableMultiColumnView2';
import ExperimentRunsTableCompactView from './ExperimentRunsTableCompactView';
import {
  LIFECYCLE_FILTER,
  MAX_DETECT_NEW_RUNS_RESULTS,
  MODEL_VERSION_FILTER,
} from './ExperimentPage';
import ExperimentViewUtil from './ExperimentViewUtil';
import DeleteRunModal from './modals/DeleteRunModal';
import RestoreRunModal from './modals/RestoreRunModal';
import { NoteInfo, NOTE_CONTENT_TAG } from '../utils/NoteUtils';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { ExperimentViewPersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
import { CSSTransition } from 'react-transition-group';
import { Spinner } from '../../common/components/Spinner';
import { RunsTableColumnSelectionDropdown } from './RunsTableColumnSelectionDropdown';
import { ColumnTypes } from '../constants';
import { getUUID } from '../../common/utils/ActionUtils';
import { ExperimentTrackingDocUrl, onboarding } from '../../common/constants';
import filterIcon from '../../common/static/filter-icon.svg';
import { StyledDropdown } from '../../common/components/StyledDropdown';
import { ExperimentNoteSection, ArtifactLocation } from './ExperimentViewHelpers';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { FlexBar } from '../../shared/building_blocks/FlexBar';
import { Button } from '../../shared/building_blocks/Button';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { SearchBox } from '../../shared/building_blocks/SearchBox';
import { Radio } from '../../shared/building_blocks/Radio';
import syncSvg from '../../common/static/sync.svg';

export const DEFAULT_EXPANDED_VALUE = false;
const { Option } = Select;
export class ExperimentView extends Component {
  constructor(props) {
    super(props);
    this.onCheckbox = this.onCheckbox.bind(this);
    this.onCompare = this.onCompare.bind(this);
    this.onDownloadCsv = this.onDownloadCsv.bind(this);
    this.onParamKeyFilterInput = this.onParamKeyFilterInput.bind(this);
    this.onMetricKeyFilterInput = this.onMetricKeyFilterInput.bind(this);
    this.onSearchInput = this.onSearchInput.bind(this);
    this.onSearch = this.onSearch.bind(this);
    this.onClear = this.onClear.bind(this);
    this.onSortBy = this.onSortBy.bind(this);
    this.onHandleSortByDropdown = this.onHandleSortByDropdown.bind(this);
    this.isAllChecked = this.isAllChecked.bind(this);
    this.onCheckbox = this.onCheckbox.bind(this);
    this.onCheckAll = this.onCheckAll.bind(this);
    this.initiateSearch = this.initiateSearch.bind(this);
    this.onDeleteRun = this.onDeleteRun.bind(this);
    this.onRestoreRun = this.onRestoreRun.bind(this);
    this.handleLifecycleFilterInput = this.handleLifecycleFilterInput.bind(this);
    this.handleModelVersionFilterInput = this.handleModelVersionFilterInput.bind(this);
    this.onCloseDeleteRunModal = this.onCloseDeleteRunModal.bind(this);
    this.onCloseRestoreRunModal = this.onCloseRestoreRunModal.bind(this);
    this.onExpand = this.onExpand.bind(this);
    this.addBagged = this.addBagged.bind(this);
    this.removeBagged = this.removeBagged.bind(this);
    this.handleSubmitEditNote = this.handleSubmitEditNote.bind(this);
    this.handleCancelEditNote = this.handleCancelEditNote.bind(this);
    this.getStartTimeColumnDisplayName = this.getStartTimeColumnDisplayName.bind(this);
    this.onHandleStartTimeDropdown = this.onHandleStartTimeDropdown.bind(this);
    const store = ExperimentView.getLocalStore(this.props.experiment.experiment_id);
    const persistedState = new ExperimentViewPersistedState(store.loadComponentState());
    const onboardingInformationStore = ExperimentView.getLocalStore(onboarding);
    this.state = {
      ...ExperimentView.getDefaultUnpersistedState(),
      persistedState: persistedState.toJSON(),
      showNotesEditor: false,
      showNotes: true,
      showFilters: false,
      showOnboardingHelper: onboardingInformationStore.getItem('showTrackingHelper') === null,
      searchInput: props.searchInput,
    };
  }
  static propTypes = {
    onSearch: PropTypes.func.isRequired,
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    modelVersionsByRunUuid: PropTypes.object.isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    history: PropTypes.any,

    // List of all parameter keys available in the runs we're viewing
    paramKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,
    // List of all metric keys available in the runs we're viewing
    metricKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,

    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(PropTypes.object).isRequired,
    // Object of experiment tags
    experimentTags: PropTypes.object.isRequired,
    // Input to the paramKeyFilter field
    paramKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,
    // Input to the paramKeyFilter field
    metricKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,

    // Input to the lifecycleFilter field
    lifecycleFilter: PropTypes.string.isRequired,
    modelVersionFilter: PropTypes.string.isRequired,

    orderByKey: PropTypes.string,
    orderByAsc: PropTypes.bool,
    startTime: PropTypes.string,

    // The initial searchInput
    searchInput: PropTypes.string.isRequired,
    searchRunsError: PropTypes.string,
    isLoading: PropTypes.bool.isRequired,
    numRunsFromLatestSearch: PropTypes.number,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    setExperimentTagApi: PropTypes.func.isRequired,

    // If child runs should be nested under their parents
    nestChildren: PropTypes.bool,
    // ML-13038: Whether to force the compact view upon page load. Used only for testing;
    // mounting ExperimentView by default will fail due to a version bug in AgGrid, so we need
    // a state-independent way of bypassing MultiColumnView.
    forceCompactTableView: PropTypes.bool,
    // The number of new runs since the last runs refresh
    numberOfNewRuns: PropTypes.number,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  /** Returns default values for state attributes that aren't persisted in local storage. */
  static getDefaultUnpersistedState() {
    return {
      // Object mapping from run UUID -> boolean (whether the run is selected)
      runsSelected: {},
      // A map { runUuid: true } of current selected child runs hidden by expander collapse
      // runsSelected + hiddenChildRunsSelected = all runs currently actually selected
      hiddenChildRunsSelected: {},
      // Text entered into the param filter field
      paramKeyFilterInput: '',
      // Text entered into the metric filter field
      metricKeyFilterInput: '',
      // Text entered into the runs-search field
      searchInput: '',
      // String error message, if any, from an attempted search
      searchErrorMessage: undefined,
      // True if a model for deleting one or more runs should be displayed
      showDeleteRunModal: false,
      // True if a model for restoring one or more runs should be displayed
      showRestoreRunModal: false,
    };
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ExperimentView component (e.g. component state such as table sort settings), for the
   * specified experiment.
   */
  static getLocalStore(experimentId) {
    return LocalStorageUtils.getStoreForComponent('ExperimentView', experimentId);
  }

  shouldComponentUpdate(nextProps, nextState) {
    // Don't update the component if a modal is showing before and after the update try.
    if (this.state.showDeleteRunModal && nextState.showDeleteRunModal) return false;
    if (this.state.showRestoreRunModal && nextState.showRestoreRunModal) return false;
    return true;
  }

  /**
   * Returns true if search filter text was updated, e.g. if a user entered new text into the
   * param filter, metric filter, or search text boxes.
   */
  filtersDidUpdate(prevState) {
    return (
      prevState.paramKeyFilterInput !== this.state.paramKeyFilterInput ||
      prevState.metricKeyFilterInput !== this.state.metricKeyFilterInput ||
      prevState.searchInput !== this.props.searchInput
    );
  }

  /** Snapshots desired attributes of the component's current state in local storage. */
  snapshotComponentState() {
    const store = ExperimentView.getLocalStore(this.props.experiment.experiment_id);
    store.saveComponentState(new ExperimentViewPersistedState(this.state.persistedState));
  }

  componentDidUpdate(prevProps, prevState) {
    // Don't snapshot state on changes to search filter text; we only want to save these on search
    // in ExperimentPage
    if (!this.filtersDidUpdate(prevState)) {
      this.snapshotComponentState();
    }
  }

  componentWillUnmount() {
    // Snapshot component state on unmounts to ensure we've captured component state in cases where
    // componentDidUpdate doesn't fire.
    this.snapshotComponentState();
  }

  componentDidMount() {
    let pageTitle = 'MLflow Experiment';
    if (this.props.experiment.name) {
      const experimentNameParts = this.props.experiment.name.split('/');
      const experimentSuffix = experimentNameParts[experimentNameParts.length - 1];
      pageTitle = `${experimentSuffix} - MLflow Experiment`;
    }
    Utils.updatePageTitle(pageTitle);
  }

  static getDerivedStateFromProps(nextProps, prevState) {
    // Compute the actual runs selected. (A run cannot be selected if it is not passed in as a prop)
    const newRunsSelected = {};
    nextProps.runInfos.forEach((rInfo) => {
      const prevRunSelected = prevState.runsSelected[rInfo.run_uuid];
      if (prevRunSelected) {
        newRunsSelected[rInfo.run_uuid] = prevRunSelected;
      }
    });
    const { paramKeyFilter, metricKeyFilter } = nextProps;
    const paramKeyFilterInput = paramKeyFilter.getFilterString();
    const metricKeyFilterInput = metricKeyFilter.getFilterString();
    return {
      ...prevState,
      paramKeyFilterInput,
      metricKeyFilterInput,
      runsSelected: newRunsSelected,
    };
  }

  setShowMultiColumns(value) {
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        showMultiColumns: value,
      }).toJSON(),
    });
  }

  disableOnboardingHelper() {
    const onboardingInformationStore = ExperimentView.getLocalStore(onboarding);
    onboardingInformationStore.setItem('showTrackingHelper', 'false');
  }

  onDeleteRun() {
    this.setState({ showDeleteRunModal: true });
  }

  onRestoreRun() {
    this.setState({ showRestoreRunModal: true });
  }

  onCloseDeleteRunModal() {
    this.setState({ showDeleteRunModal: false });
  }

  onCloseRestoreRunModal() {
    this.setState({ showRestoreRunModal: false });
  }

  /**
   * Mark a column as bagged by removing it from the appropriate array of unbagged columns.
   * @param isParam If true, the column is assumed to be a metric column; if false, the column is
   *                assumed to be a param column.
   * @param colName Name of the column (metric or param key).
   */
  addBagged(isParam, colName) {
    const unbagged = isParam
      ? this.state.persistedState.unbaggedParams
      : this.state.persistedState.unbaggedMetrics;
    const idx = unbagged.indexOf(colName);
    const newUnbagged =
      idx >= 0 ? unbagged.slice(0, idx).concat(unbagged.slice(idx + 1, unbagged.length)) : unbagged;
    const stateKey = isParam ? 'unbaggedParams' : 'unbaggedMetrics';
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        [stateKey]: newUnbagged,
      }).toJSON(),
    });
  }

  /**
   * Mark a column as unbagged by adding it to the appropriate array of unbagged columns.
   * @param isParam If true, the column is assumed to be a metric column; if false, the column is
   *                assumed to be a param column.
   * @param colName Name of the column (metric or param key).
   */
  removeBagged(isParam, colName) {
    const unbagged = isParam
      ? this.state.persistedState.unbaggedParams
      : this.state.persistedState.unbaggedMetrics;
    const stateKey = isParam ? 'unbaggedParams' : 'unbaggedMetrics';
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        [stateKey]: unbagged.concat([colName]),
      }).toJSON(),
    });
  }

  handleSubmitEditNote(note) {
    const { experiment_id } = this.props.experiment;
    this.props
      .setExperimentTagApi(experiment_id, NOTE_CONTENT_TAG, note, getUUID())
      .then(() => this.setState({ showNotesEditor: false }));
  }

  handleCancelEditNote() {
    this.setState({ showNotesEditor: false });
  }

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showNotesEditor: true });
  };

  handleColumnSelectionCheck = (categorizedUncheckedKeys) => {
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        categorizedUncheckedKeys,
      }).toJSON(),
    });
  };

  handleFilterToggle = () => {
    this.setState((previousState) => ({ showFilters: !previousState.showFilters }));
  };

  getFilteredKeys(keyList, columnType) {
    const { categorizedUncheckedKeys } = this.state.persistedState;
    return _.difference(keyList, categorizedUncheckedKeys[columnType]);
  }

  renderOnboardingContent() {
    const learnMoreLinkUrl = ExperimentView.getLearnMoreLinkUrl();
    const content = (
      <div>
        <FormattedMessage
          // eslint-disable-next-line max-len
          defaultMessage='Track machine learning training runs in an experiment. <link>Learn more</link>'
          // eslint-disable-next-line max-len
          description='Information banner text to provide more information about experiments runs page'
          values={{
            link: (chunks) => (
              <a
                href={learnMoreLinkUrl}
                target='_blank'
                rel='noopener noreferrer'
                className='LinkColor'
              >
                {chunks}
              </a>
            ),
          }}
        />
      </div>
    );

    return this.state.showOnboardingHelper ? (
      <Alert
        className='information'
        description={content}
        type='info'
        showIcon
        closable
        onClose={() => this.disableOnboardingHelper()}
      />
    ) : null;
  }

  static getLearnMoreLinkUrl = () => ExperimentTrackingDocUrl;

  getModelVersionMenuItem(key, data_test_id, text) {
    return (
      <Menu.Item
        data-test-id={data_test_id}
        active={this.props.modelVersionFilter === key}
        onSelect={this.handleModelVersionFilterInput}
        key={key}
      >
        {text}
      </Menu.Item>
    );
  }

  getStartTimeColumnDisplayName() {
    return {
      ALL: this.props.intl.formatMessage({
        defaultMessage: 'All',
        description: 'Option for the start select dropdown to render all runs',
      }),
      LAST_HOUR: this.props.intl.formatMessage({
        defaultMessage: 'Last hour',
        description: 'Option for the start select dropdown to filter runs from the last hour',
      }),
      LAST_24_HOURS: this.props.intl.formatMessage({
        defaultMessage: 'Last 24 hours',
        description: 'Option for the start select dropdown to filter runs from the last 24 hours',
      }),
      LAST_7_DAYS: this.props.intl.formatMessage({
        defaultMessage: 'Last 7 days',
        description: 'Option for the start select dropdown to filter runs from the last 7 days',
      }),
      LAST_30_DAYS: this.props.intl.formatMessage({
        defaultMessage: 'Last 30 days',
        description: 'Option for the start select dropdown to filter runs from the last 30 days',
      }),
      LAST_YEAR: this.props.intl.formatMessage({
        defaultMessage: 'Last year',
        description: 'Option for the start select dropdown to filter runs since the last 1 year',
      }),
    };
  }

  render() {
    const {
      runInfos,
      isLoading,
      loadingMore,
      numRunsFromLatestSearch,
      handleLoadMoreRuns,
      experimentTags,
      experiment,
      tagsList,
      paramKeyList,
      metricKeyList,
      orderByKey,
      orderByAsc,
      startTime,
      nestChildren,
      numberOfNewRuns,
    } = this.props;
    const { experiment_id, name } = experiment;
    const { persistedState } = this.state;
    const { unbaggedParams, unbaggedMetrics, categorizedUncheckedKeys } = persistedState;
    const filteredParamKeys = this.getFilteredKeys(paramKeyList, ColumnTypes.PARAMS);
    const filteredMetricKeys = this.getFilteredKeys(metricKeyList, ColumnTypes.METRICS);
    const visibleTagKeyList = Utils.getVisibleTagKeyList(tagsList);
    const filteredVisibleTagKeyList = this.getFilteredKeys(visibleTagKeyList, ColumnTypes.TAGS);
    const filteredUnbaggedParamKeys = this.getFilteredKeys(unbaggedParams, ColumnTypes.PARAMS);
    const filteredUnbaggedMetricKeys = this.getFilteredKeys(unbaggedMetrics, ColumnTypes.METRICS);
    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    const deleteDisabled = Object.keys(this.state.runsSelected).length < 1;
    const restoreDisabled = Object.keys(this.state.runsSelected).length < 1;
    const noteInfo = NoteInfo.fromTags(experimentTags);
    const startTimeColumnLabels = this.getStartTimeColumnDisplayName();
    const searchInputHelpTooltipContent = (
      <div className='search-input-tooltip-content'>
        <FormattedMessage
          defaultMessage='Search runs using a simplified version of the SQL <b>WHERE</b> clause'
          description='Tooltip string to explain how to search runs from the experiments table'
        />
        <br />
        <FormattedMessage
          defaultMessage='<link>Learn more</link>'
          // eslint-disable-next-line max-len
          description='Learn more tooltip link to learn more on how to search in an experiments run table'
          values={{
            link: (chunks) => (
              <a
                href='https://www.mlflow.org/docs/latest/search-syntax.html'
                target='_blank'
                rel='noopener noreferrer'
              >
                {chunks}
              </a>
            ),
          }}
        />
      </div>
    );
    const experimentRunsState = (lifecycleFilter) => {
      return lifecycleFilter === LIFECYCLE_FILTER.ACTIVE
        ? this.props.intl.formatMessage({
            defaultMessage: 'Active',
            description: 'Linked model dropdown option to show active experiment runs',
          })
        : this.props.intl.formatMessage({
            defaultMessage: 'Deleted',
            description: 'Linked model dropdown option to show deleted experiment runs',
          });
    };
    /* eslint-disable prefer-const */
    let breadcrumbs = [];
    let form;

    let title = <>{name}</>;

    const artifactLocationProps = {
      experiment: this.props.experiment,
      intl: this.props.intl,
    };

    const {
      ColumnSortByAscending,
      ColumnSortByDescending,
      SortDelimiterSymbol,
    } = ExperimentViewUtil;
    const ColumnSortByOrder = [ColumnSortByAscending, ColumnSortByDescending];

    return (
      <div className='ExperimentView runs-table-flex-container'>
        <DeleteRunModal
          isOpen={this.state.showDeleteRunModal}
          onClose={this.onCloseDeleteRunModal}
          selectedRunIds={Object.keys(this.state.runsSelected)}
        />
        <RestoreRunModal
          isOpen={this.state.showRestoreRunModal}
          onClose={this.onCloseRestoreRunModal}
          selectedRunIds={Object.keys(this.state.runsSelected)}
        />
        <PageHeader title={title} copyText={name} breadcrumbs={breadcrumbs} feedbackForm={form} />
        {this.renderOnboardingContent()}
        <Descriptions className='metadata-list'>
          <Descriptions.Item
            label={this.props.intl.formatMessage({
              defaultMessage: 'Experiment ID',
              description: 'Label for displaying the current experiment in view',
            })}
          >
            {experiment_id}
          </Descriptions.Item>
          <ArtifactLocation {...artifactLocationProps} />
        </Descriptions>
        <div className='ExperimentView-info'>
          <ExperimentNoteSection
            noteInfo={noteInfo}
            handleCancelEditNote={this.handleCancelEditNote}
            handleSubmitEditNote={this.handleSubmitEditNote}
            showNotesEditor={this.state.showNotesEditor}
            startEditingDescription={this.startEditingDescription}
          />
        </div>
        <div className='ExperimentView-runs runs-table-flex-container'>
          {this.props.searchRunsError ? (
            <div className='error-message'>
              <span className='error-message'>{this.props.searchRunsError}</span>
            </div>
          ) : null}
          <Spacer size='medium'>
            <div>
              <FormattedMessage
                // eslint-disable-next-line max-len
                defaultMessage='Showing {length} matching {length, plural, =0 {runs} =1 {run} other {runs}}'
                // eslint-disable-next-line max-len
                description='Message for displaying how many runs match search criteria on experiment page'
                values={{ length: runInfos.length }}
              />
            </div>
            <FlexBar
              left={
                <Spacer size='small' direction='horizontal'>
                  <Badge
                    count={numberOfNewRuns}
                    offset={[-5, 5]}
                    style={{ backgroundColor: '#33804D' }}
                    overflowCount={MAX_DETECT_NEW_RUNS_RESULTS - 1}
                  >
                    <Button className='refresh-button' onClick={this.initiateSearch}>
                      <img alt='' title='Refresh runs' src={syncSvg} height={24} width={24} />
                      <FormattedMessage
                        defaultMessage='Refresh'
                        description='refresh button text to refresh the experiment runs'
                      />
                    </Button>
                  </Badge>
                  <Button
                    className='compare-button'
                    disabled={compareDisabled}
                    onClick={this.onCompare}
                  >
                    <FormattedMessage
                      defaultMessage='Compare'
                      // eslint-disable-next-line max-len
                      description='String for the compare button to compare experiment runs to find an ideal model'
                    />
                  </Button>
                  {this.props.lifecycleFilter === LIFECYCLE_FILTER.ACTIVE ? (
                    <Button
                      className='delete-restore-button'
                      disabled={deleteDisabled}
                      onClick={this.onDeleteRun}
                    >
                      <FormattedMessage
                        defaultMessage='Delete'
                        // eslint-disable-next-line max-len
                        description='String for the delete button to delete a particular experiment run'
                      />
                    </Button>
                  ) : null}
                  {this.props.lifecycleFilter === LIFECYCLE_FILTER.DELETED ? (
                    <Button disabled={restoreDisabled} onClick={this.onRestoreRun}>
                      <FormattedMessage
                        defaultMessage='Restore'
                        // eslint-disable-next-line max-len
                        description='String for the restore button to undo the experiments that were deleted'
                      />
                    </Button>
                  ) : null}
                  <Button className='csv-button' onClick={this.onDownloadCsv}>
                    <FormattedMessage
                      defaultMessage='Download CSV'
                      // eslint-disable-next-line max-len
                      description='String for the download csv button to download experiments offline in a CSV format'
                    />
                    <i className='fas fa-download' />
                  </Button>
                  <Tooltip
                    title={this.props.intl.formatMessage({
                      defaultMessage: 'Sort by',
                      description:
                        'Sort label for the sort select dropdown for experiment runs view',
                    })}
                  >
                    <Select
                      className='sort-select'
                      value={
                        orderByKey
                          ? `${orderByKey}${SortDelimiterSymbol}${
                              orderByAsc ? ColumnSortByAscending : ColumnSortByDescending
                            }`
                          : this.props.intl.formatMessage({
                              defaultMessage: 'Sort by',
                              description:
                                // eslint-disable-next-line max-len
                                'Sort by default option for sort by select dropdown for experiment runs',
                            })
                      }
                      size='large'
                      onChange={this.onHandleSortByDropdown}
                      data-test-id='sort-select-dropdown'
                    >
                      {Object.keys(ExperimentViewUtil.AttributeColumnSortLabel).reduce(
                        (sortOptions, sortLabelKey) => {
                          const sortLabel =
                            ExperimentViewUtil.AttributeColumnSortLabel[sortLabelKey];
                          if (
                            !categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES].includes(sortLabel)
                          ) {
                            ColumnSortByOrder.forEach((order) => {
                              sortOptions.push(
                                <Option
                                  key={sortLabel}
                                  title={sortLabel}
                                  data-test-id={`sort-select-${sortLabel}-${order}`}
                                  value={
                                    ExperimentViewUtil.AttributeColumnSortKey[sortLabelKey] +
                                    SortDelimiterSymbol +
                                    order
                                  }
                                >
                                  {order === ColumnSortByAscending ? (
                                    <Icon type='arrow-up' />
                                  ) : (
                                    <Icon type='arrow-down' />
                                  )}{' '}
                                  {sortLabel}
                                </Option>,
                              );
                            });
                          }

                          return sortOptions;
                        },
                        [],
                      )}
                      {filteredMetricKeys.reduce((sortOptions, metricKey) => {
                        ColumnSortByOrder.forEach((order) => {
                          sortOptions.push(
                            <Option
                              key={metricKey}
                              title={metricKey}
                              data-test-id={`sort-select-${metricKey}-${order}`}
                              value={`${ExperimentViewUtil.makeCanonicalKey(
                                ColumnTypes.METRICS,
                                metricKey,
                              )}${SortDelimiterSymbol}${order}`}
                            >
                              {order === ColumnSortByAscending ? (
                                <Icon type='arrow-up' />
                              ) : (
                                <Icon type='arrow-down' />
                              )}{' '}
                              {metricKey}
                            </Option>,
                          );
                        });

                        return sortOptions;
                      }, [])}
                      {filteredParamKeys.reduce((sortOptions, paramKey) => {
                        ColumnSortByOrder.forEach((order) => {
                          sortOptions.push(
                            <Option
                              key={paramKey}
                              title={paramKey}
                              data-test-id={`sort-select-${paramKey}-${order}`}
                              value={`${ExperimentViewUtil.makeCanonicalKey(
                                ColumnTypes.PARAMS,
                                paramKey,
                              )}${SortDelimiterSymbol}${order}`}
                            >
                              {order === ColumnSortByAscending ? (
                                <Icon type='arrow-up' />
                              ) : (
                                <Icon type='arrow-down' />
                              )}{' '}
                              {paramKey}
                            </Option>,
                          );
                        });

                        return sortOptions;
                      }, [])}
                    </Select>
                  </Tooltip>
                  <Tooltip
                    title={this.props.intl.formatMessage({
                      defaultMessage: 'Start time',
                      description:
                        'Label for the start time select dropdown for experiment runs view',
                    })}
                  >
                    <Select
                      className='start-time-select'
                      value={startTime}
                      size='large'
                      onChange={this.onHandleStartTimeDropdown}
                      data-test-id='start-time-select-dropdown'
                    >
                      {Object.keys(startTimeColumnLabels).map((startTimeKey) => (
                        <Option
                          key={startTimeKey}
                          title={startTimeColumnLabels[startTimeKey]}
                          data-test-id={`start-time-select-${startTimeKey}`}
                          value={startTimeKey}
                        >
                          {startTimeColumnLabels[startTimeKey]}
                        </Option>
                      ))}
                    </Select>
                  </Tooltip>
                </Spacer>
              }
              right={
                <Spacer size='large' direction='horizontal'>
                  <Spacer size='medium' direction='horizontal'>
                    <Radio
                      defaultValue={
                        this.state.persistedState.showMultiColumns ? 'gridView' : 'compactView'
                      }
                      items={[
                        {
                          value: 'compactView',
                          itemContent: <i className={'fas fa-list'} />,
                          onClick: (e) => this.setShowMultiColumns(false),
                          dataTestId: 'compact-runs-table-view-button',
                        },
                        {
                          value: 'gridView',
                          itemContent: <i className={'fas fa-table'} />,
                          onClick: (e) => this.setShowMultiColumns(true),
                          dataTestId: 'detailed-runs-table-view-button',
                        },
                      ]}
                    />
                    <RunsTableColumnSelectionDropdown
                      paramKeyList={paramKeyList}
                      metricKeyList={metricKeyList}
                      visibleTagKeyList={visibleTagKeyList}
                      categorizedUncheckedKeys={categorizedUncheckedKeys}
                      onCheck={this.handleColumnSelectionCheck}
                    />
                  </Spacer>
                  <Spacer direction='horizontal' size='small'>
                    <Popover
                      overlayClassName='search-input-tooltip'
                      content={searchInputHelpTooltipContent}
                      placement='bottom'
                    >
                      <Icon
                        type='question-circle'
                        className='ExperimentView-search-help'
                        theme='filled'
                      />
                    </Popover>
                    <div style={styles.searchBox}>
                      <SearchBox
                        onChange={this.onSearchInput}
                        value={this.state.searchInput}
                        onSearch={this.onSearch}
                        placeholder='metrics.rmse < 1 and params.model = "tree"'
                      />
                    </div>
                    <Button dataTestId='filter-button' onClick={this.handleFilterToggle}>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <img className='filterIcon' src={filterIcon} alt='Filter' />
                        <FormattedMessage
                          defaultMessage='Filter'
                          // eslint-disable-next-line max-len
                          description='String for the filter button to filter experiment runs table which match the search criteria'
                        />
                      </div>
                    </Button>
                    <Button dataTestId='clear-button' onClick={this.onClear}>
                      <FormattedMessage
                        defaultMessage='Clear'
                        // eslint-disable-next-line max-len
                        description='String for the clear button to clear any filters or sorting that we may have applied on the experiment table'
                      />
                    </Button>
                  </Spacer>
                </Spacer>
              }
            />
            <CSSTransition
              in={this.state.showFilters}
              timeout={300}
              classNames='lifecycleButtons'
              unmountOnExit
            >
              <div className='ExperimentView-lifecycle-input'>
                <div className='filter-wrapper' style={styles.lifecycleButtonFilterWrapper}>
                  <FormattedMessage
                    defaultMessage='State:'
                    // eslint-disable-next-line max-len
                    description='Filtering label to filter experiments based on state of active or deleted'
                  />
                  <StyledDropdown
                    key={this.props.lifecycleFilter}
                    title={experimentRunsState(this.props.lifecycleFilter)}
                    dropdownOptions={
                      <Menu onClick={this.handleLifecycleFilterInput}>
                        <Menu.Item
                          data-test-id='active-runs-menu-item'
                          active={this.props.lifecycleFilter === LIFECYCLE_FILTER.ACTIVE}
                          key={LIFECYCLE_FILTER.ACTIVE}
                        >
                          {experimentRunsState(LIFECYCLE_FILTER.ACTIVE)}
                        </Menu.Item>
                        <Menu.Item
                          data-test-id='deleted-runs-menu-item'
                          active={this.props.lifecycleFilter === LIFECYCLE_FILTER.DELETED}
                          key={LIFECYCLE_FILTER.DELETED}
                        >
                          {experimentRunsState(LIFECYCLE_FILTER.DELETED)}
                        </Menu.Item>
                      </Menu>
                    }
                    triggers={['click']}
                    id='ExperimentView-lifecycle-button-id'
                    className='ExperimentView-lifecycle-button'
                  />
                  <span className='model-versions-label'>
                    <FormattedMessage
                      defaultMessage='Linked Models:'
                      // eslint-disable-next-line max-len
                      description='Filtering label for filtering experiments based on if the models are linked or not to the experiment'
                    />
                  </span>
                  <StyledDropdown
                    key={this.props.modelVersionFilter}
                    title={this.props.modelVersionFilter}
                    dropdownOptions={
                      <Menu onClick={this.handleModelVersionFilterInput}>
                        {this.getModelVersionMenuItem(
                          MODEL_VERSION_FILTER.ALL_RUNS,
                          'all-runs-menu-item',
                          this.props.intl.formatMessage({
                            defaultMessage: 'All Runs',
                            description: 'Linked model dropdown option to show all experiment runs',
                          }),
                        )}
                        {this.getModelVersionMenuItem(
                          MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS,
                          'model-versions-runs-menu-item',
                          this.props.intl.formatMessage({
                            defaultMessage: 'With Model Versions',
                            description:
                              // eslint-disable-next-line max-len
                              'Linked model dropdown option to show experiment runs with model versions only',
                          }),
                        )}
                        {this.getModelVersionMenuItem(
                          MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS,
                          'no-model-versions-runs-menu-item',
                          this.props.intl.formatMessage({
                            defaultMessage: 'Without Model Versions',
                            description:
                              // eslint-disable-next-line max-len
                              'Linked model dropdown option to show experiment runs without model versions only',
                          }),
                        )}
                      </Menu>
                    }
                    triggers={['click']}
                    className='ExperimentView-linked-model-button'
                    id='ExperimentView-linked-model-button-id'
                  />
                </div>
              </div>
            </CSSTransition>
            {this.state.persistedState.showMultiColumns && !this.props.forceCompactTableView ? (
              <ExperimentRunsTableMultiColumnView2
                experimentId={experiment.experiment_id}
                modelVersionsByRunUuid={this.props.modelVersionsByRunUuid}
                onSelectionChange={this.handleMultiColumnViewSelectionChange}
                runInfos={this.props.runInfos}
                paramsList={this.props.paramsList}
                metricsList={this.props.metricsList}
                tagsList={this.props.tagsList}
                paramKeyList={filteredParamKeys}
                metricKeyList={filteredMetricKeys}
                visibleTagKeyList={filteredVisibleTagKeyList}
                categorizedUncheckedKeys={categorizedUncheckedKeys}
                isAllChecked={this.isAllChecked()}
                onSortBy={this.onSortBy}
                orderByKey={orderByKey}
                orderByAsc={this.props.orderByAsc}
                runsSelected={this.state.runsSelected}
                runsExpanded={this.state.persistedState.runsExpanded}
                onExpand={this.onExpand}
                numRunsFromLatestSearch={numRunsFromLatestSearch}
                handleLoadMoreRuns={handleLoadMoreRuns}
                loadingMore={loadingMore}
                isLoading={isLoading}
                nestChildren={nestChildren}
              />
            ) : isLoading ? (
              <Spinner showImmediately />
            ) : (
              <ExperimentRunsTableCompactView
                onCheckbox={this.onCheckbox}
                runInfos={this.props.runInfos}
                modelVersionsByRunUuid={this.props.modelVersionsByRunUuid}
                // Bagged param and metric keys
                paramKeyList={filteredParamKeys}
                metricKeyList={filteredMetricKeys}
                paramsList={this.props.paramsList}
                metricsList={this.props.metricsList}
                tagsList={this.props.tagsList}
                categorizedUncheckedKeys={categorizedUncheckedKeys}
                onCheck={this.handleColumnSelectionCheck}
                onCheckAll={this.onCheckAll}
                isAllChecked={this.isAllChecked()}
                onSortBy={this.onSortBy}
                orderByKey={orderByKey}
                orderByAsc={this.props.orderByAsc}
                runsSelected={this.state.runsSelected}
                runsExpanded={this.state.persistedState.runsExpanded}
                onExpand={this.onExpand}
                unbaggedMetrics={filteredUnbaggedMetricKeys}
                unbaggedParams={filteredUnbaggedParamKeys}
                onAddBagged={this.addBagged}
                onRemoveBagged={this.removeBagged}
                numRunsFromLatestSearch={numRunsFromLatestSearch}
                handleLoadMoreRuns={handleLoadMoreRuns}
                loadingMore={loadingMore}
                nestChildren={nestChildren}
              />
            )}
          </Spacer>
        </div>
      </div>
    );
  }

  onHandleSortByDropdown(value) {
    const [orderByKey, orderBy] = value.split(ExperimentViewUtil.SortDelimiterSymbol);

    this.onSortBy(orderByKey, orderBy === ExperimentViewUtil.ColumnSortByAscending);
  }

  onHandleStartTimeDropdown(startTime) {
    this.initiateSearch({ startTime });
  }

  onSortBy(orderByKey, orderByAsc) {
    this.initiateSearch({ orderByKey, orderByAsc });
  }

  initiateSearch({
    paramKeyFilterInput,
    metricKeyFilterInput,
    searchInput,
    lifecycleFilterInput,
    modelVersionFilterInput,
    orderByKey,
    orderByAsc,
    startTime,
  }) {
    const myParamKeyFilterInput =
      paramKeyFilterInput !== undefined ? paramKeyFilterInput : this.state.paramKeyFilterInput;
    const myMetricKeyFilterInput =
      metricKeyFilterInput !== undefined ? metricKeyFilterInput : this.state.metricKeyFilterInput;
    const mySearchInput = searchInput !== undefined ? searchInput : this.props.searchInput;
    const myLifecycleFilterInput =
      lifecycleFilterInput !== undefined ? lifecycleFilterInput : this.props.lifecycleFilter;
    const myOrderByKey = orderByKey !== undefined ? orderByKey : this.props.orderByKey;
    const myOrderByAsc = orderByAsc !== undefined ? orderByAsc : this.props.orderByAsc;
    const myModelVersionFilterInput = modelVersionFilterInput || this.props.modelVersionFilter;
    const myStartTime = startTime || this.props.startTime;
    try {
      this.props.onSearch(
        myParamKeyFilterInput,
        myMetricKeyFilterInput,
        mySearchInput,
        myLifecycleFilterInput,
        myOrderByKey,
        myOrderByAsc,
        myModelVersionFilterInput,
        myStartTime,
      );
    } catch (ex) {
      if (ex.errorMessage !== undefined) {
        this.setState({ searchErrorMessage: ex.errorMessage });
      } else {
        throw ex;
      }
    }
  }

  onCheckbox(runUuid) {
    const newState = Object.assign({}, this.state);
    if (this.state.runsSelected[runUuid]) {
      delete newState.runsSelected[runUuid];
      this.setState(newState);
    } else {
      this.setState({
        runsSelected: {
          ...this.state.runsSelected,
          [runUuid]: true,
        },
      });
    }
  }

  isAllChecked() {
    return Object.keys(this.state.runsSelected).length === this.props.runInfos.length;
  }

  onCheckAll() {
    if (this.isAllChecked()) {
      this.setState({ runsSelected: {} });
    } else {
      const runsSelected = {};
      this.props.runInfos.forEach(({ run_uuid }) => {
        runsSelected[run_uuid] = true;
      });
      this.setState({ runsSelected: runsSelected });
    }
  }

  // Special handler for ag-grid selection change event from multi-column view
  handleMultiColumnViewSelectionChange = (selectedRunUuids) => {
    const runsSelected = {};
    selectedRunUuids.forEach((runUuid) => (runsSelected[runUuid] = true));
    this.setState({ runsSelected });
  };

  onExpand(runId, childRunIds) {
    const { runsSelected, hiddenChildRunsSelected, persistedState } = this.state;
    const { runsExpanded } = persistedState;
    const expandedAfterToggle = !ExperimentViewUtil.isExpanderOpen(runsExpanded, runId);
    const newRunsSelected = { ...runsSelected };
    const newHiddenChildRunsSelected = { ...hiddenChildRunsSelected };

    if (expandedAfterToggle) {
      // User expanded current run, to automatically select previous hidden child runs that were
      // selected, find them in `hiddenChildRunsSelected` and add them to `newRunsSelected`
      childRunIds.forEach((childRunId) => {
        if (hiddenChildRunsSelected[childRunId]) {
          delete newHiddenChildRunsSelected[childRunId];
          newRunsSelected[childRunId] = true;
        }
      });
    } else {
      // User collapsed current run, find all currently selected child runs from `runsSelected` and
      // save them to `newHiddenChildRunsSelected`
      childRunIds.forEach((childRunId) => {
        if (runsSelected[childRunId]) {
          delete newRunsSelected[childRunId];
          newHiddenChildRunsSelected[childRunId] = true;
        }
      });
    }

    this.setState({
      runsSelected: newRunsSelected,
      hiddenChildRunsSelected: newHiddenChildRunsSelected,
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        runsExpanded: {
          ...this.state.persistedState.runsExpanded,
          [runId]: expandedAfterToggle,
        },
      }).toJSON(),
    });
  }

  onParamKeyFilterInput(event) {
    this.setState({ paramKeyFilterInput: event.target.value });
  }

  onMetricKeyFilterInput(event) {
    this.setState({ metricKeyFilterInput: event.target.value });
  }

  onSearchInput(event) {
    this.setState({ searchInput: event.target.value });
  }

  handleLifecycleFilterInput({ key: lifecycleFilterInput }) {
    this.initiateSearch({ lifecycleFilterInput });
  }

  handleModelVersionFilterInput({ key: modelVersionFilterInput }) {
    this.initiateSearch({ modelVersionFilterInput });
  }

  onSearch(e, searchInput) {
    if (e !== undefined) {
      e.preventDefault();
    }
    const { paramKeyFilterInput, metricKeyFilterInput } = this.state;
    this.initiateSearch({
      paramKeyFilterInput: paramKeyFilterInput,
      metricKeyFilterInput: metricKeyFilterInput,
      searchInput: searchInput,
    });
  }

  onClear() {
    // When user clicks "Clear", preserve multicolumn toggle state but reset other persisted state
    // attributes to their default values.
    const newPersistedState = new ExperimentViewPersistedState({
      showMultiColumns: this.state.persistedState.showMultiColumns,
    });
    this.setState({ persistedState: newPersistedState.toJSON(), searchInput: '' }, () => {
      this.snapshotComponentState();
      this.initiateSearch({
        paramKeyFilterInput: '',
        metricKeyFilterInput: '',
        searchInput: '',
        lifecycleFilterInput: LIFECYCLE_FILTER.ACTIVE,
        modelVersionFilterInput: MODEL_VERSION_FILTER.ALL_RUNS,
        orderByKey: null,
        orderByAsc: false,
        startTime: 'ALL',
      });
    });
  }

  onCompare() {
    const runsSelectedList = Object.keys(this.state.runsSelected);
    this.props.history.push(
      Routes.getCompareRunPageRoute(runsSelectedList, this.props.experiment.getExperimentId()),
    );
  }

  onDownloadCsv() {
    const { paramKeyList, metricKeyList, runInfos, paramsList, metricsList, tagsList } = this.props;
    const filteredParamKeys = this.getFilteredKeys(paramKeyList, ColumnTypes.PARAMS);
    const filteredMetricKeys = this.getFilteredKeys(metricKeyList, ColumnTypes.METRICS);
    const visibleTagKeys = Utils.getVisibleTagKeyList(tagsList);
    const filteredTagKeys = this.getFilteredKeys(visibleTagKeys, ColumnTypes.TAGS);
    const csv = ExperimentView.runInfosToCsv(
      runInfos,
      filteredParamKeys,
      filteredMetricKeys,
      filteredTagKeys,
      paramsList,
      metricsList,
      tagsList,
    );
    const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
    saveAs(blob, 'runs.csv');
  }

  /**
   * Format a string for insertion into a CSV file.
   */
  static csvEscape(str) {
    if (str === undefined) {
      return '';
    }
    if (/[,"\r\n]/.test(str)) {
      return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
  }

  /**
   * Convert a table to a CSV string.
   *
   * @param columns Names of columns
   * @param data Array of rows, each of which are an array of field values
   */
  static tableToCsv(columns, data) {
    let csv = '';
    let i;

    for (i = 0; i < columns.length; i++) {
      csv += ExperimentView.csvEscape(columns[i]);
      if (i < columns.length - 1) {
        csv += ',';
      }
    }
    csv += '\n';

    for (i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        csv += ExperimentView.csvEscape(data[i][j]);
        if (j < data[i].length - 1) {
          csv += ',';
        }
      }
      csv += '\n';
    }

    return csv;
  }

  /**
   * Convert an array of run infos to a CSV string, extracting the params and metrics in the
   * provided lists.
   */
  static runInfosToCsv(
    runInfos,
    paramKeyList,
    metricKeyList,
    tagKeyList,
    paramsList,
    metricsList,
    tagsList,
  ) {
    const columns = [
      'Run ID',
      'Name',
      'Source Type',
      'Source Name',
      'User',
      'Status',
      ...paramKeyList,
      ...metricKeyList,
      ...tagKeyList,
    ];

    const data = runInfos.map((runInfo, index) => {
      const row = [
        runInfo.run_uuid,
        Utils.getRunName(tagsList[index]), // add run name to csv export row
        Utils.getSourceType(tagsList[index]),
        Utils.getSourceName(tagsList[index]),
        Utils.getUser(runInfo, tagsList[index]),
        runInfo.status,
      ];
      const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[index]);
      const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[index]);
      const tagsMap = tagsList[index];

      paramKeyList.forEach((paramKey) => {
        if (paramsMap[paramKey]) {
          row.push(paramsMap[paramKey].getValue());
        } else {
          row.push('');
        }
      });
      metricKeyList.forEach((metricKey) => {
        if (metricsMap[metricKey]) {
          row.push(metricsMap[metricKey].getValue());
        } else {
          row.push('');
        }
      });
      tagKeyList.forEach((tagKey) => {
        if (tagsMap[tagKey]) {
          row.push(tagsMap[tagKey].getValue());
        } else {
          row.push('');
        }
      });
      return row;
    });

    return ExperimentView.tableToCsv(columns, data);
  }
}

export const mapStateToProps = (state, ownProps) => {
  const { lifecycleFilter, modelVersionFilter } = ownProps;

  // The runUuids we should serve.
  const { runInfosByUuid } = state.entities;
  const runUuids = Object.values(runInfosByUuid)
    .filter((r) => r.experiment_id === ownProps.experimentId.toString())
    .map((r) => r.run_uuid);

  const { modelVersionsByRunUuid } = state.entities;

  const runInfos = runUuids
    .map((run_id) => getRunInfo(run_id, state))
    .filter((rInfo) => {
      if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
        return rInfo.lifecycle_stage === 'active';
      } else {
        return rInfo.lifecycle_stage === 'deleted';
      }
    })
    .filter((rInfo) => {
      if (modelVersionFilter === MODEL_VERSION_FILTER.ALL_RUNS) {
        return true;
      } else if (modelVersionFilter === MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS) {
        return rInfo.run_uuid in modelVersionsByRunUuid;
      } else if (modelVersionFilter === MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS) {
        return !(rInfo.run_uuid in modelVersionsByRunUuid);
      } else {
        console.warn('Invalid input to model version filter - defaulting to showing all runs.');
        return true;
      }
    });
  const metricKeysSet = new Set();
  const paramKeysSet = new Set();
  const metricsList = runInfos.map((runInfo) => {
    const metricsByRunUuid = getLatestMetrics(runInfo.getRunUuid(), state);
    const metrics = Object.values(metricsByRunUuid || {});
    metrics.forEach((metric) => {
      metricKeysSet.add(metric.key);
    });
    return metrics;
  });
  const paramsList = runInfos.map((runInfo) => {
    const params = Object.values(getParams(runInfo.getRunUuid(), state));
    params.forEach((param) => {
      paramKeysSet.add(param.key);
    });
    return params;
  });

  const tagsList = runInfos.map((runInfo) => getRunTags(runInfo.getRunUuid(), state));
  const experimentTags = getExperimentTags(ownProps.experimentId, state);
  return {
    runInfos,
    modelVersionsByRunUuid,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
    metricsList,
    paramsList,
    tagsList,
    experimentTags,
  };
};

const mapDispatchToProps = {
  setExperimentTagApi,
};

const styles = {
  lifecycleButtonFilterWrapper: {
    marginLeft: '48px',
  },
  searchBox: {
    width: '446px',
  },
};

export const ExperimentViewWithIntl = injectIntl(ExperimentView);
export default withRouter(connect(mapStateToProps, mapDispatchToProps)(ExperimentViewWithIntl));
