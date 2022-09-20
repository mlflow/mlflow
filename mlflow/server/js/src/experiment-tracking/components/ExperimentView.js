/* eslint-disable max-lines */
import React, { Component } from 'react';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { injectIntl, FormattedMessage } from 'react-intl';
// eslint-disable-next-line no-unused-vars
import { Link, withRouter } from 'react-router-dom';
import { Alert, Badge, Descriptions, Menu, Tooltip, Switch } from 'antd';
import {
  Popover,
  Typography,
  useDesignSystemTheme,
  Select,
  Button,
  ArrowDownIcon,
  ArrowUpIcon,
  SyncIcon,
  FilterIcon,
  DownloadIcon,
  QuestionMarkFillIcon,
} from '@databricks/design-system';

import './ExperimentView.css';
import { getExperimentTags, getParams, getRunInfo, getRunTags } from '../reducers/Reducers';
import { setExperimentTagApi } from '../actions';
import Routes from '../routes';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import { ExperimentRunsTableMultiColumnView2 } from './ExperimentRunsTableMultiColumnView2';
import ExperimentViewUtil from './ExperimentViewUtil';
import DeleteRunModal from './modals/DeleteRunModal';
import RestoreRunModal from './modals/RestoreRunModal';
import { GetLinkModal } from './modals/GetLinkModal';
import { NoteInfo, NOTE_CONTENT_TAG } from '../utils/NoteUtils';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { ExperimentViewPersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
import { CSSTransition } from 'react-transition-group';
import { Spinner } from '../../common/components/Spinner';
import { RunsTableColumnSelectionDropdown } from './RunsTableColumnSelectionDropdown';
import { getUUID } from '../../common/utils/ActionUtils';
import {
  ExperimentSearchSyntaxDocUrl,
  ExperimentTrackingDocUrl,
  onboarding,
} from '../../common/constants';
import { StyledDropdown } from '../../common/components/StyledDropdown';
import { ExperimentNoteSection, ArtifactLocation } from './ExperimentViewHelpers';
import { OverflowMenu, PageHeader, HeaderButton } from '../../shared/building_blocks/PageHeader';
import { FlexBar } from '../../shared/building_blocks/FlexBar';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { SearchBox } from '../../shared/building_blocks/SearchBox';
import { middleTruncateStr } from '../../common/utils/StringUtils';
import {
  COLUMN_TYPES,
  LIFECYCLE_FILTER,
  MAX_DETECT_NEW_RUNS_RESULTS,
  MODEL_VERSION_FILTER,
  ATTRIBUTE_COLUMN_SORT_LABEL,
  ATTRIBUTE_COLUMN_SORT_KEY,
  COLUMN_SORT_BY_ASC,
  COLUMN_SORT_BY_DESC,
  SORT_DELIMITER_SYMBOL,
} from '../constants';

function RefreshBadge({ children, count }) {
  const { theme } = useDesignSystemTheme();
  return (
    <Badge
      count={count}
      offset={[-5, 5]}
      // Note: Must use style over classNames, otherwise you get weird antd behavior
      style={{ backgroundColor: theme.colors.lime, zIndex: 1 }}
      overflowCount={MAX_DETECT_NEW_RUNS_RESULTS - 1}
    >
      {children}
    </Badge>
  );
}

RefreshBadge.propTypes = {
  children: PropTypes.node.isRequired,
  count: PropTypes.number.isRequired,
};

export const DEFAULT_EXPANDED_VALUE = false;
const { Option } = Select;
const { Text } = Typography;
export class ExperimentView extends Component {
  constructor(props) {
    super(props);
    this.onSortBy = this.onSortBy.bind(this);
    this.onHandleSortByDropdown = this.onHandleSortByDropdown.bind(this);
    this.initiateSearch = this.initiateSearch.bind(this);
    this.onDeleteRun = this.onDeleteRun.bind(this);
    this.onRestoreRun = this.onRestoreRun.bind(this);
    this.handleLifecycleFilterInput = this.handleLifecycleFilterInput.bind(this);
    this.handleModelVersionFilterInput = this.handleModelVersionFilterInput.bind(this);
    this.onCloseDeleteRunModal = this.onCloseDeleteRunModal.bind(this);
    this.onCloseRestoreRunModal = this.onCloseRestoreRunModal.bind(this);
    this.onExpand = this.onExpand.bind(this);
    this.handleSubmitEditNote = this.handleSubmitEditNote.bind(this);
    this.handleCancelEditNote = this.handleCancelEditNote.bind(this);
    this.getStartTimeColumnDisplayName = this.getStartTimeColumnDisplayName.bind(this);
    this.onHandleStartTimeDropdown = this.onHandleStartTimeDropdown.bind(this);
    this.handleDiffSwitchChange = this.handleDiffSwitchChange.bind(this);
    this.handleShareButtonClick = this.handleShareButtonClick.bind(this);
    const store = ExperimentView.getLocalStore(this.stringifyExperimentIds());
    const persistedState = new ExperimentViewPersistedState({
      ...store.loadComponentState(),
    });
    const onboardingInformationStore = ExperimentView.getLocalStore(onboarding);
    this.state = {
      ...ExperimentView.getDefaultUnpersistedState(),
      persistedState: persistedState.toJSON(),
      showNotesEditor: false,
      showNotes: true,
      showFilters: false,
      showOnboardingHelper: onboardingInformationStore.getItem('showTrackingHelper') === null,
      searchInput: props.searchInput,
      lastExperimentIds: undefined,
      showGetLinkModal: false,
    };
  }
  static propTypes = {
    compareExperiments: PropTypes.bool,
    onSearch: PropTypes.func.isRequired,
    onClear: PropTypes.func.isRequired,
    setShowMultiColumns: PropTypes.func.isRequired,
    handleColumnSelectionCheck: PropTypes.func.isRequired,
    handleDiffSwitchChange: PropTypes.func.isRequired,
    updateUrlWithViewState: PropTypes.func.isRequired,
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    modelVersionsByRunUuid: PropTypes.object.isRequired,
    experiments: PropTypes.arrayOf(PropTypes.instanceOf(Experiment)).isRequired,
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
    // The initial searchInput
    searchInput: PropTypes.string.isRequired,
    orderByKey: PropTypes.string.isRequired,
    orderByAsc: PropTypes.bool.isRequired,
    startTime: PropTypes.string.isRequired,
    lifecycleFilter: PropTypes.string.isRequired,
    modelVersionFilter: PropTypes.string.isRequired,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
    diffSwitchSelected: PropTypes.bool.isRequired,
    preSwitchCategorizedUncheckedKeys: PropTypes.object.isRequired,
    postSwitchCategorizedUncheckedKeys: PropTypes.object.isRequired,

    searchRunsError: PropTypes.string,
    isLoading: PropTypes.bool.isRequired,
    nextPageToken: PropTypes.string,
    numRunsFromLatestSearch: PropTypes.number,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    setExperimentTagApi: PropTypes.func.isRequired,
    // If child runs should be nested under their parents
    nestChildren: PropTypes.bool,
    // The number of new runs since the last runs refresh
    numberOfNewRuns: PropTypes.number,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  static defaultProps = {
    compareExperiments: false,
  };

  /** Returns default values for state attributes that aren't persisted in local storage. */
  static getDefaultUnpersistedState() {
    return {
      // Object mapping from run UUID -> boolean (whether the run is selected)
      runsSelected: {},
      // A map { runUuid: true } of current selected child runs hidden by expander collapse
      // runsSelected + hiddenChildRunsSelected = all runs currently actually selected
      hiddenChildRunsSelected: {},
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
   * specified id.
   */
  static getLocalStore(id) {
    return LocalStorageUtils.getStoreForComponent('ExperimentView', id);
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
    return prevState.searchInput !== this.props.searchInput;
  }

  stringifyExperimentIds() {
    return JSON.stringify(this.props.experiments.map(({ experiment_id }) => experiment_id).sort());
  }

  /** Snapshots desired attributes of the component's current state in local storage. */
  snapshotComponentState() {
    const store = ExperimentView.getLocalStore(this.stringifyExperimentIds());
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
    if (!this.props.compareExperiments && this.props.experiments[0].name) {
      const experimentNameParts = this.props.experiments[0].name.split('/');
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
    let persistedState;
    let lastExperimentIds;
    let newPersistedState = {};
    // Reported during ESLint upgrade
    // eslint-disable-next-line react/prop-types
    if (!_.isEqual(nextProps.experimentIds, prevState.lastExperimentIds)) {
      persistedState =
        prevState.lastExperimentIds === undefined
          ? prevState.persistedState
          : new ExperimentViewPersistedState().toJSON();
      // Reported during ESLint upgrade
      // eslint-disable-next-line react/prop-types
      lastExperimentIds = nextProps.experimentIds;
      newPersistedState = { persistedState, lastExperimentIds };
    }
    return {
      ...prevState,
      ...newPersistedState,
      runsSelected: newRunsSelected,
    };
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

  handleSubmitEditNote(note) {
    const { experiment_id } = this.props.experiments[0];
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

  handleFilterToggle = () => {
    this.setState((previousState) => ({ showFilters: !previousState.showFilters }));
  };

  getFilteredKeys(keyList, columnType) {
    const { categorizedUncheckedKeys } = this.props;
    return _.difference(keyList, categorizedUncheckedKeys[columnType]);
  }

  renderOnboardingContent() {
    const learnMoreLinkUrl = ExperimentView.getLearnMoreLinkUrl();
    const content = (
      <div>
        <FormattedMessage
          // eslint-disable-next-line max-len
          defaultMessage='Track machine learning training runs in experiments. <link>Learn more</link>'
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
        css={styles.alert}
        message={content}
        type='info'
        showIcon
        closable
        onClose={() => this.disableOnboardingHelper()}
      />
    ) : null;
  }

  getUrl() {
    return window.location.href;
  }

  renderGetLinkModal() {
    const { showGetLinkModal } = this.state;
    return (
      <GetLinkModal
        link={this.getUrl()}
        visible={showGetLinkModal}
        onCancel={() => this.setState({ showGetLinkModal: false })}
      />
    );
  }

  handleShareButtonClick() {
    const { updateUrlWithViewState } = this.props;
    updateUrlWithViewState();
    this.setState({ showGetLinkModal: true });
  }

  renderShareButton() {
    return (
      <HeaderButton
        type='secondary'
        onClick={this.handleShareButtonClick}
        data-test-id='share-button'
      >
        <FormattedMessage
          defaultMessage='Share'
          description='Text for share button on experiment view page header'
        />
      </HeaderButton>
    );
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
        defaultMessage: 'All time',
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

  getExperimentOverflowItems() {
    const menuItems = [];
    return menuItems;
  }

  getCompareExperimentsPageTitle() {
    return this.props.intl.formatMessage(
      {
        defaultMessage: 'Displaying Runs from {numExperiments} Experiments',
        description: 'Message shown when displaying runs from multiple experiments',
      },
      {
        numExperiments: this.props.experiments.length,
      },
    );
  }

  render() {
    const {
      runInfos,
      isLoading,
      loadingMore,
      nextPageToken,
      numRunsFromLatestSearch,
      handleLoadMoreRuns,
      experimentTags,
      experiments,
      tagsList,
      paramKeyList,
      metricKeyList,
      orderByKey,
      orderByAsc,
      startTime,
      categorizedUncheckedKeys,
      diffSwitchSelected,
      nestChildren,
      numberOfNewRuns,
    } = this.props;
    const { experiment_id, name } = experiments[0];
    const filteredParamKeys = this.getFilteredKeys(paramKeyList, COLUMN_TYPES.PARAMS);
    const filteredMetricKeys = this.getFilteredKeys(metricKeyList, COLUMN_TYPES.METRICS);
    const visibleTagKeyList = Utils.getVisibleTagKeyList(tagsList);
    const filteredVisibleTagKeyList = this.getFilteredKeys(visibleTagKeyList, COLUMN_TYPES.TAGS);
    const restoreDisabled = Object.keys(this.state.runsSelected).length < 1;
    const noteInfo = NoteInfo.fromTags(experimentTags);
    const startTimeColumnLabels = this.getStartTimeColumnDisplayName();
    const searchInputHelpTooltipContent = (
      <div className='search-input-tooltip-content'>
        <FormattedMessage
          defaultMessage='Search runs using a simplified version of the SQL {whereBold} clause'
          description='Tooltip string to explain how to search runs from the experiments table'
          values={{ whereBold: <b>WHERE</b> }}
        />
        <br />
        <FormattedMessage
          defaultMessage='<link>Learn more</link>'
          // eslint-disable-next-line max-len
          description='Learn more tooltip link to learn more on how to search in an experiments run table'
          values={{
            link: (chunks) => (
              <a href={ExperimentSearchSyntaxDocUrl} target='_blank' rel='noopener noreferrer'>
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

    const artifactLocationProps = {
      experiment: this.props.experiments[0],
      intl: this.props.intl,
    };

    const ColumnSortByOrder = [COLUMN_SORT_BY_ASC, COLUMN_SORT_BY_DESC];
    let sortOptions = [];
    const attributesSortBy = Object.keys(ATTRIBUTE_COLUMN_SORT_LABEL).reduce(
      (options, sortLabelKey) => {
        const sortLabel = ATTRIBUTE_COLUMN_SORT_LABEL[sortLabelKey];
        if (!categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES].includes(sortLabel)) {
          ColumnSortByOrder.forEach((order) => {
            options.push({
              label: sortLabel,
              value: ATTRIBUTE_COLUMN_SORT_KEY[sortLabelKey] + SORT_DELIMITER_SYMBOL + order,
              order,
            });
          });
        }

        return options;
      },
      [],
    );
    const metricsSortBy = filteredMetricKeys.reduce((options, sortLabelKey) => {
      ColumnSortByOrder.forEach((order) => {
        options.push({
          label: sortLabelKey,
          value: `${ExperimentViewUtil.makeCanonicalKey(
            COLUMN_TYPES.METRICS,
            sortLabelKey,
          )}${SORT_DELIMITER_SYMBOL}${order}`,
          order,
        });
      });

      return options;
    }, []);
    const paramsSortBy = filteredParamKeys.reduce((options, sortLabelKey) => {
      ColumnSortByOrder.forEach((order) => {
        options.push({
          label: sortLabelKey,
          value: `${ExperimentViewUtil.makeCanonicalKey(
            COLUMN_TYPES.PARAMS,
            sortLabelKey,
          )}${SORT_DELIMITER_SYMBOL}${order}`,
          order,
        });
      });

      return options;
    }, []);
    sortOptions = [...attributesSortBy, ...metricsSortBy, ...paramsSortBy];

    const pageHeaderTitle = this.props.compareExperiments ? (
      this.getCompareExperimentsPageTitle()
    ) : (
      <>
        {name}
        <Text
          size='xl'
          dangerouslySetAntdProps={{
            copyable: {
              text: name,
              tooltips: [
                this.props.intl.formatMessage({
                  defaultMessage: 'Copy',
                  description:
                    'Copy tooltip to copy experiment name from experiment runs table header',
                }),
              ],
            },
          }}
        />
      </>
    );

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
        {this.renderGetLinkModal()}
        <PageHeader title={pageHeaderTitle} breadcrumbs={breadcrumbs}>
          {!this.props.compareExperiments && (
            <OverflowMenu
              data-test-id='experiment-view-page-header'
              menu={this.getExperimentOverflowItems()}
            />
          )}
          {this.renderShareButton()}
        </PageHeader>
        {this.renderOnboardingContent()}
        {!this.props.compareExperiments && (
          <>
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
          </>
        )}
        <div className='ExperimentView-runs runs-table-flex-container'>
          {this.props.searchRunsError ? (
            <div className='error-message'>
              <span className='error-message'>{this.props.searchRunsError}</span>
            </div>
          ) : null}
          <Spacer size='medium'>
            <FlexBar
              left={
                <Spacer size='small' direction='horizontal'>
                  <div css={styles.controlBar}>
                    <RefreshBadge count={numberOfNewRuns}>
                      <Button
                        onClick={this.initiateSearch}
                        className='refresh-button'
                        icon={<SyncIcon />}
                      >
                        <FormattedMessage
                          defaultMessage='Refresh'
                          description='refresh button text to refresh the experiment runs'
                        />
                      </Button>
                    </RefreshBadge>
                    <Button
                      disabled={Object.keys(this.state.runsSelected).length < 2}
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
                        disabled={Object.keys(this.state.runsSelected).length < 1}
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
                    <Button onClick={this.onDownloadCsv} icon={<DownloadIcon />}>
                      <FormattedMessage
                        defaultMessage='Download CSV'
                        // eslint-disable-next-line max-len
                        description='String for the download csv button to download experiments offline in a CSV format'
                      />
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
                            ? `${orderByKey}${SORT_DELIMITER_SYMBOL}${
                                orderByAsc ? COLUMN_SORT_BY_ASC : COLUMN_SORT_BY_DESC
                              }`
                            : this.props.intl.formatMessage({
                                defaultMessage: 'Sort by',
                                description:
                                  // eslint-disable-next-line max-len
                                  'Sort by default option for sort by select dropdown for experiment runs',
                              })
                        }
                        // Temporarily we're disabling virtualized list to maintain
                        // backwards compatiblity. Functional unit tests rely heavily
                        // on non-virtualized values.
                        dangerouslySetAntdProps={{ virtual: false }}
                        onChange={this.onHandleSortByDropdown}
                        data-test-id='sort-select-dropdown'
                        dropdownStyle={{ minWidth: '30%' }}
                      >
                        {sortOptions.map((sortOption) => (
                          <Option
                            key={sortOption.value}
                            title={sortOption.label}
                            data-test-id={`sort-select-${sortOption.label}-${sortOption.order}`}
                            value={sortOption.value}
                          >
                            {sortOption.order === COLUMN_SORT_BY_ASC ? (
                              <ArrowUpIcon />
                            ) : (
                              <ArrowDownIcon />
                            )}{' '}
                            {middleTruncateStr(sortOption.label, 50)}
                          </Option>
                        ))}
                      </Select>
                    </Tooltip>
                    <Tooltip
                      title={this.props.intl.formatMessage({
                        defaultMessage: 'Created during',
                        description:
                          'Label for the start time select dropdown for experiment runs view',
                      })}
                    >
                      <Select
                        className='start-time-select'
                        value={startTime}
                        onChange={this.onHandleStartTimeDropdown}
                        data-test-id='start-time-select-dropdown'
                        // Temporarily we're disabling virtualized list to maintain
                        // backwards compatiblity. Functional unit tests rely heavily
                        // on non-virtualized values.
                        dangerouslySetAntdProps={{ virtual: false }}
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
                  </div>
                </Spacer>
              }
              right={
                <div css={styles.controlBar}>
                  <Spacer size='large' direction='horizontal'>
                    <RunsTableColumnSelectionDropdown
                      paramKeyList={paramKeyList}
                      metricKeyList={metricKeyList}
                      visibleTagKeyList={visibleTagKeyList}
                      categorizedUncheckedKeys={categorizedUncheckedKeys}
                      onCheck={this.props.handleColumnSelectionCheck}
                    />
                    <Spacer size='small' direction='horizontal'>
                      {this.props.intl.formatMessage({
                        defaultMessage: 'Only show differences',
                        description:
                          'Switch to select only columns with different values across runs',
                      })}
                      <Tooltip
                        title={this.props.intl.formatMessage({
                          defaultMessage: 'Only show columns with differences',
                          description:
                            'Switch to select only columns with different values across runs',
                        })}
                      >
                        <Switch
                          css={styles.columnSwitch}
                          // dataTestId='diff-switch'
                          checked={diffSwitchSelected}
                          onChange={this.handleDiffSwitchChange}
                        />
                      </Tooltip>
                    </Spacer>
                    <Spacer direction='horizontal' size='small'>
                      <Popover
                        overlayClassName='search-input-tooltip'
                        content={searchInputHelpTooltipContent}
                        placement='bottom'
                      >
                        <QuestionMarkFillIcon className='ExperimentView-search-help' />
                      </Popover>
                      <div css={styles.searchBox}>
                        <SearchBox
                          onChange={this.onSearchInput}
                          value={this.state.searchInput}
                          onSearch={this.onSearch}
                          placeholder='metrics.rmse < 1 and params.model = "tree"'
                        />
                      </div>
                      <Button
                        data-testid='filter-button'
                        onClick={this.handleFilterToggle}
                        icon={<FilterIcon />}
                      >
                        <FormattedMessage
                          defaultMessage='Filter'
                          // eslint-disable-next-line max-len
                          description='String for the filter button to filter experiment runs table which match the search criteria'
                        />
                      </Button>
                      <Button data-test-id='clear-button' onClick={this.onClear}>
                        <FormattedMessage
                          defaultMessage='Clear'
                          // eslint-disable-next-line max-len
                          description='String for the clear button to clear any filters or sorting that we may have applied on the experiment table'
                        />
                      </Button>
                    </Spacer>
                  </Spacer>
                </div>
              }
            />
            <CSSTransition
              in={this.state.showFilters}
              timeout={300}
              classNames='lifecycleButtons'
              unmountOnExit
            >
              <div className='ExperimentView-lifecycle-input'>
                <div className='filter-wrapper' css={styles.lifecycleButtonFilterWrapper}>
                  <FormattedMessage
                    defaultMessage='State:'
                    // eslint-disable-next-line max-len
                    description='Filtering label to filter experiments based on state of active or deleted'
                  />
                  <StyledDropdown
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
            <div>
              <FormattedMessage
                // eslint-disable-next-line max-len
                defaultMessage='Showing {length} matching {length, plural, =0 {runs} =1 {run} other {runs}}'
                // eslint-disable-next-line max-len
                description='Message for displaying how many runs match search criteria on experiment page'
                values={{ length: runInfos.length }}
              />
            </div>
            {isLoading ? (
              <Spinner showImmediately />
            ) : (
              <ExperimentRunsTableMultiColumnView2
                compareExperiments={this.props.compareExperiments}
                experiments={experiments}
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
                orderByAsc={orderByAsc}
                runsSelected={this.state.runsSelected}
                runsExpanded={this.state.persistedState.runsExpanded}
                onExpand={this.onExpand}
                nextPageToken={nextPageToken}
                numRunsFromLatestSearch={numRunsFromLatestSearch}
                handleLoadMoreRuns={handleLoadMoreRuns}
                loadingMore={loadingMore}
                isLoading={isLoading}
                nestChildren={nestChildren}
              />
            )}
          </Spacer>
        </div>
      </div>
    );
  }

  onHandleSortByDropdown(value) {
    const [orderByKey, orderBy] = value.split(SORT_DELIMITER_SYMBOL);

    this.onSortBy(orderByKey, orderBy === COLUMN_SORT_BY_ASC);
  }

  onHandleStartTimeDropdown(startTime) {
    this.initiateSearch({ startTime });
  }

  onSortBy(orderByKey, orderByAsc) {
    this.initiateSearch({ orderByKey, orderByAsc });
  }

  initiateSearch(value) {
    try {
      this.props.onSearch(value);
    } catch (ex) {
      if (ex.errorMessage !== undefined) {
        this.setState({ searchErrorMessage: ex.errorMessage });
      } else {
        throw ex;
      }
    }
  }

  onCheckbox = (runUuid) => {
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
  };

  isAllChecked = () => {
    return Object.keys(this.state.runsSelected).length === this.props.runInfos.length;
  };

  onCheckAll = () => {
    if (this.isAllChecked()) {
      this.setState({ runsSelected: {} });
    } else {
      const runsSelected = {};
      this.props.runInfos.forEach(({ run_uuid }) => {
        runsSelected[run_uuid] = true;
      });
      this.setState({ runsSelected: runsSelected });
    }
  };

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

  onSearchInput = (event) => {
    this.setState({ searchInput: event.target.value });
  };

  handleLifecycleFilterInput({ key: lifecycleFilter }) {
    this.initiateSearch({ lifecycleFilter });
  }

  handleModelVersionFilterInput({ key: modelVersionFilter }) {
    this.initiateSearch({ modelVersionFilter });
  }

  handleDiffSwitchChange = () => {
    let newCategorizedUncheckedKeys;
    let switchPersistedState;
    if (!this.props.diffSwitchSelected) {
      // When turning on the diff switch
      const { categorizedUncheckedKeys } = this.props;
      newCategorizedUncheckedKeys = ExperimentViewUtil.getCategorizedUncheckedKeysDiffView({
        ...this.props,
        categorizedUncheckedKeys,
      });
      switchPersistedState = {
        preSwitchCategorizedUncheckedKeys: categorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys: newCategorizedUncheckedKeys,
      };
    } else {
      // When turning off the diff switch
      const {
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        categorizedUncheckedKeys: currCategorizedUncheckedKeys,
      } = this.props;
      newCategorizedUncheckedKeys = ExperimentViewUtil.getRestoredCategorizedUncheckedKeys({
        preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys,
        currCategorizedUncheckedKeys,
      });
      switchPersistedState = {};
    }

    this.props.handleDiffSwitchChange({
      categorizedUncheckedKeys: newCategorizedUncheckedKeys,
      ...switchPersistedState,
    });
  };

  onSearch = (e, searchInput) => {
    if (e !== undefined) {
      e.preventDefault();
    }
    this.initiateSearch({
      searchInput: searchInput,
    });
  };

  onClear = () => {
    this.setState(
      {
        searchInput: '',
      },
      () => {
        this.props.onClear();
      },
    );
  };

  onCompare = () => {
    const { runInfos } = this.props;
    const runsSelectedList = Object.keys(this.state.runsSelected);
    const experimentIds = runInfos
      .filter(({ run_uuid }) => runsSelectedList.includes(run_uuid))
      .map(({ experiment_id }) => experiment_id);
    this.props.history.push(
      Routes.getCompareRunPageRoute(runsSelectedList, [...new Set(experimentIds)].sort()),
    );
  };

  onDownloadCsv = () => {
    const { paramKeyList, metricKeyList, runInfos, paramsList, metricsList, tagsList } = this.props;
    const filteredParamKeys = this.getFilteredKeys(paramKeyList, COLUMN_TYPES.PARAMS);
    const filteredMetricKeys = this.getFilteredKeys(metricKeyList, COLUMN_TYPES.METRICS);
    const visibleTagKeys = Utils.getVisibleTagKeyList(tagsList);
    const filteredTagKeys = this.getFilteredKeys(visibleTagKeys, COLUMN_TYPES.TAGS);
    const csv = ExperimentViewUtil.runInfosToCsv(
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
  };
}

export const mapStateToProps = (state, ownProps) => {
  const { lifecycleFilter, modelVersionFilter } = ownProps;

  // The runUuids we should serve.
  const { runInfosByUuid } = state.entities;
  const experimentIds = ownProps.experiments.map(({ experiment_id }) => experiment_id.toString());
  const runUuids = Object.values(runInfosByUuid)
    .filter(({ experiment_id }) => experimentIds.includes(experiment_id))
    .map(({ run_uuid }) => run_uuid);

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
  // Only show description if we're viewing runs from a single experiment
  const experimentTags =
    !ownProps.compareExperiments && getExperimentTags(ownProps.experiments[0].experiment_id, state);
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
  controlBar: (theme) => ({
    display: 'flex',
    gap: theme.spacing.sm,
  }),
  searchBox: {
    width: '446px',
  },
  alert: {
    marginBottom: 16,
    padding: 16,
    background: '#edfafe' /* Gray-background */,
    border: '1px solid #eeeeee',
    boxShadow: '0px 1px 2px rgba(0, 0, 0, 0.12)' /* Dropshadow */,
    borderRadius: 4,
  },
  displaySwitch: (theme) => ({ svg: { width: theme.spacing.md, height: theme.spacing.md } }),
  columnSwitch: { margin: '5px' },
  filterButtonWrapper: { display: 'flex', alignItems: 'center' },
};

export const ExperimentViewWithIntl = injectIntl(ExperimentView);
export default withRouter(connect(mapStateToProps, mapDispatchToProps)(ExperimentViewWithIntl));
