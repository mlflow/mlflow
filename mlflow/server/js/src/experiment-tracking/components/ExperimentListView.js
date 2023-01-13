import React, { Component } from 'react';
import PropTypes from 'prop-types';
import 'react-virtualized/styles.css';
import { List as VList, AutoSizer, InfiniteLoader } from 'react-virtualized';
import { List } from 'antd';
import {
  Input,
  Typography,
  CaretDownSquareIcon,
  PlusCircleBorderIcon,
  PencilIcon,
  Checkbox,
  SearchIcon,
  WithDesignSystemThemeHoc,
} from '@databricks/design-system';
import { css } from '@emotion/react';
import { Link, withRouter } from 'react-router-dom';
import {
  experimentListSearchInput,
  searchExperimentsApi,
  loadMoreExperimentsApi,
} from '../actions';
import {
  getExperimentListSearchInput,
  getExperimentListPreviousSearchInput,
  getLoadMoreExperimentsNextPageToken,
  getExperimentsFiltered,
} from '../reducers/Reducers';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';

export class ExperimentListView extends Component {
  static propTypes = {
    activeExperimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    history: PropTypes.object.isRequired,
    dispatchSearchExperimentsApi: PropTypes.func.isRequired,
    dispatchLoadMoreExperiemntsApi: PropTypes.func.isRequired,
    dispatchSearchInput: PropTypes.func.isRequired,
    experiments: PropTypes.arrayOf(PropTypes.object).isRequired,
    searchInput: PropTypes.string.isRequired,
    previousSearchInput: PropTypes.string.isRequired,
    nextPageToken: PropTypes.string,
    designSystemThemeApi: PropTypes.shape({ theme: PropTypes.object }).isRequired,
  };

  constructor(props) {
    super(props);

    this.state = {
      hidden: false,
      showCreateExperimentModal: false,
      showDeleteExperimentModal: false,
      showRenameExperimentModal: false,
      selectedExperimentId: '0',
      selectedExperimentName: '',
      checkedKeys: this.props.activeExperimentIds,
      innerHeight: 90,
    };
  }

  bindListRef = (ref) => {
    this.list = ref;
  };

  infiniteLoaderRef = (ref) => {
    this.infiteLoader = ref;
  };

  componentDidUpdate = () => {
    if (this.list) {
      this.list.forceUpdateGrid();
    }
  };

  handleSearchInputChange = (event) => {
    if (event.target.value === '') {
      // Dispatch twice to totally clear it
      this.props.dispatchSearchInput(event.target.value);
      // Need another page token for ''
      this.props.dispatchLoadMoreExperiemntsApi({});
      this.infiteLoader.resetLoadMoreRowsCache();
    }
    this.props.dispatchSearchInput(event.target.value);
  };

  handleLoadMore = () => {
    let params = {};
    const currentInput = this.props.searchInput;
    const previousInput = this.props.previousSearchInput;
    // Leave the filter off if we have a blank string.
    if (currentInput !== '') {
      params = { ...params, filter: `name LIKE '%${currentInput}%'` };
    }
    // Use a next page if the input was not altered to get more.
    if (currentInput === previousInput) {
      params = { ...params, pageToken: this.props.nextPageToken };
    } else {
      // Needed if the input has changed.
      this.infiteLoader.resetLoadMoreRowsCache();
    }

    // Need to dispatch both to make the prior and current match on
    // the next round through.
    this.props.dispatchSearchInput(currentInput);
    this.props.dispatchLoadMoreExperiemntsApi(params);
  };

  onScroll = ({ startIndex, stopIndex }) => {
    const isScrolledToLastItem = stopIndex >= this.props.experiments.length;
    if (!isScrolledToLastItem) {
      return;
    }
    this.handleLoadMore();
  };

  updateSelectedExperiment = (experimentId, experimentName) => {
    this.setState({
      selectedExperimentId: experimentId,
      selectedExperimentName: experimentName,
    });
  };

  handleCreateExperiment = () => {
    this.setState({
      showCreateExperimentModal: true,
    });
  };

  handleDeleteExperiment = (experimentId, experimentName) => () => {
    this.setState({
      showDeleteExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleRenameExperiment = (experimentId, experimentName) => () => {
    this.setState({
      showRenameExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleCloseCreateExperimentModal = () => {
    this.setState({
      showCreateExperimentModal: false,
    });
  };

  handleCloseDeleteExperimentModal = () => {
    this.setState({
      showDeleteExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  handleCloseRenameExperimentModal = () => {
    this.setState({
      showRenameExperimentModal: false,
    });
    // reset
    this.updateSelectedExperiment('0', '');
  };

  pushExperimentRoute = () => {
    if (this.state.checkedKeys.length > 0) {
      const route =
        this.state.checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(this.state.checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(this.state.checkedKeys);
      this.props.history.push(route);
    }
  };

  // Add a key if it does not exist, remove it if it does
  // Always keep at least one experiment checked if it is only the active one.
  handleCheck = (e, key) => {
    this.setState((prevState, props) => {
      let { checkedKeys } = prevState;
      if (e === true && !prevState.checkedKeys.includes(key)) {
        checkedKeys = [key, ...prevState.checkedKeys];
      }
      if (e === false && props.activeExperimentIds.length !== 1) {
        checkedKeys = prevState.checkedKeys.filter((i) => i !== key);
      }
      return { checkedKeys: checkedKeys };
    }, this.pushExperimentRoute);
  };

  // Avoid calling emotion for every list item
  theme = this.props.theme;
  activeExperimentListItem = classNames.getExperimentListItemContainer(
    true,
    this.props.designSystemThemeApi.theme,
  );
  inactiveExperimentListItem = classNames.getExperimentListItemContainer(
    false,
    this.props.designSystemThemeApi.theme,
  );

  renderListItem = ({ index, key, style, isScrolling }) => {
    const item = this.props.experiments[index];
    const { activeExperimentIds } = this.props;
    const { checkedKeys } = this.state;
    const isActive = activeExperimentIds.includes(item.experiment_id);
    const isChecked = checkedKeys.includes(item.experiment_id);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
    // Clicking the link removes all checks and marks other experiments
    // as not active.
    return (
      <div
        css={isActive ? this.activeExperimentListItem : this.inactiveExperimentListItem}
        data-test-id={dataTestId}
        key={key}
        style={style}
      >
        <List.Item
          key={item.experiment_id}
          bordered='false'
          prefixCls={'experiment-list-meta'}
          css={classNames.experimentListItem}
          actions={[
            <Checkbox
              id={item.experiment_id}
              key={item.experiment_id}
              onChange={(e) => this.handleCheck(e, item.experiment_id)}
              checked={isChecked || isActive}
              data-test-id='experiment-list-item-check-box'
            ></Checkbox>,
            <Link
              className={'experiment-link'}
              to={Routes.getExperimentPageRoute(item.experiment_id)}
              onClick={() => this.setState({ checkedKeys: [item.experiment_id] })}
              title={item.name}
              data-test-id='experiment-list-item-link'
            >
              {item.name}
            </Link>,
            <IconButton
              icon={<PencilIcon />}
              onClick={this.handleRenameExperiment(item.experiment_id, item.name)}
              data-test-id='rename-experiment-button'
              css={classNames.renameExperiment}
            />,
            <IconButton
              icon={<i className='far fa-trash-o' />}
              onClick={this.handleDeleteExperiment(item.experiment_id, item.name)}
              css={classNames.deleteExperiment}
              data-test-id='delete-experiment-button'
            />,
          ]}
        ></List.Item>
      </div>
    );
  };

  isRowLoaded = ({ index }) => {
    return !!this.props.experiments[index];
  };

  noRowsRenderer = () => {
    return <Typography.Text>No experiments match search input</Typography.Text>;
  };

  showExperimentList = () => this.setState({ hidden: false });
  hideExperimentList = () => this.setState({ hidden: true });

  render() {
    const { activeExperimentIds, experiments, searchInput } = this.props;
    const { hidden } = this.state;

    if (hidden) {
      return (
        <CaretDownSquareIcon
          rotate={-90}
          onClick={this.showExperimentList}
          css={{ fontSize: '24px' }}
          title='Show experiment list'
        />
      );
    }

    return (
      <div id='experiment-list-outer-container' css={classNames.experimentListOuterContainer}>
        <CreateExperimentModal
          isOpen={this.state.showCreateExperimentModal}
          onClose={this.handleCloseCreateExperimentModal}
        />
        <DeleteExperimentModal
          isOpen={this.state.showDeleteExperimentModal}
          onClose={this.handleCloseDeleteExperimentModal}
          activeExperimentIds={activeExperimentIds}
          experimentId={this.state.selectedExperimentId}
          experimentName={this.state.selectedExperimentName}
        />
        <RenameExperimentModal
          isOpen={this.state.showRenameExperimentModal}
          onClose={this.handleCloseRenameExperimentModal}
          experimentId={this.state.selectedExperimentId}
          experimentName={this.state.selectedExperimentName}
        />
        <div css={classNames.experimentTitleContainer}>
          <Typography.Title level={2} css={classNames.experimentTitle}>
            Experiments
          </Typography.Title>
          <PlusCircleBorderIcon
            onClick={this.handleCreateExperiment}
            css={{
              fontSize: '24px',
              marginLeft: 'auto',
            }}
            title='New Experiment'
            data-test-id='create-experiment-button'
          />
          <CaretDownSquareIcon
            onClick={this.hideExperimentList}
            rotate={90}
            css={{ fontSize: '24px' }}
            title='Hide experiment list'
          />
        </div>
        <div css={classNames.experimentSearchContainer}>
          <Input
            placeholder='Search Experiments'
            aria-label='search experiments'
            value={searchInput}
            onChange={this.handleSearchInputChange}
            onPressEnter={this.handleLoadMore}
            data-test-id='search-experiment-input'
            css={classNames.experimentSearchInput}
          />
          <SearchIcon
            onClick={this.handleLoadMore}
            title='Search/refresh experiments list'
            css={classNames.experimentSearchIcon}
          />
        </div>
        <div>
          <AutoSizer>
            {({ width, height }) => (
              <InfiniteLoader
                isRowLoaded={this.isRowLoaded}
                loadMoreRows={this.onScroll}
                rowCount={Number.MAX_SAFE_INTEGER} // arbitrarily high value
                ref={this.infiniteLoaderRef}
              >
                {({ onRowsRendered, registerChild }) => (
                  <VList
                    rowRenderer={this.renderListItem}
                    data={this.state.checkedKeys}
                    onRowsRendered={onRowsRendered}
                    noRowsRenderer={this.noRowsRenderer}
                    ref={this.bindListRef}
                    rowHeight={32}
                    overscanRowCount={10}
                    height={height}
                    width={width}
                    rowCount={experiments.length}
                  />
                )}
              </InfiniteLoader>
            )}
          </AutoSizer>
        </div>
      </div>
    );
  }
}

const classNames = {
  experimentListOuterContainer: css({
    boxSizing: 'border-box',
    marginTop: '24px',
    marginLeft: '24px',
    marginRight: '8px',
    paddingRight: '16px',
    width: '100%',
    // Ensure it displays experiment names for smaller screens, but don't
    // take more than 20% of the screen.
    minWidth: 'max(280px, 20vw)',
    maxWidth: '20vw',
    display: 'grid',
    gridTemplateRows: 'auto auto 1fr',
  }),
  experimentTitleContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  experimentTitle: {
    margin: 0,
  },
  experimentSearchContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '2px',
  },
  experimentSearchInput: {
    margin: 0,
    flex: '6 1 0',
  },
  experimentSearchIcon: {
    fontSize: '24px',
    marginLeft: 'auto',
    flex: '1 1 0',
  },
  experimentListContainer: {
    marginTop: '12px',
  },
  getExperimentListItemContainer: (isActive, theme) =>
    css({
      display: 'flex',
      marginRight: '8px',
      paddingRight: '5px',
      borderLeft: isActive ? `solid ${theme.colors.primary}` : 'solid transparent',
      borderLeftWidth: 4,
      backgroundColor: isActive ? theme.colors.actionDefaultBackgroundPress : 'transparent',
    }),
  experimentListItem: css({
    display: 'grid',
    // Make the items line up
    width: '100%',
    '.experiment-list-meta-item-action': {
      display: 'grid',
      gridTemplateColumns: 'auto 1fr auto auto',
      paddingLeft: '0px',
      marginBottom: '4px',
      marginTop: '4px',
      li: {
        paddingRight: '4px',
        paddingLeft: '4px',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        textAlign: 'left',
        fontSize: '13px',
      },
    },
  }),
  renameExperiment: {
    justifySelf: 'end',
  },
  // Use a larger margin to avoid overlapping the vertical scrollbar
  deleteExperiment: {
    justifySelf: 'end',
    marginRight: '10px',
  },
};

const mapStateToProps = (state) => {
  // const allExperiments = getExperiments(state);
  const searchInput = getExperimentListSearchInput(state);
  const previousSearchInput = getExperimentListPreviousSearchInput(state);
  const experiments = getExperimentsFiltered(state);
  const nextPageToken = getLoadMoreExperimentsNextPageToken(state);
  return { experiments, nextPageToken, searchInput, previousSearchInput };
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchSearchExperimentsApi: (params) => {
      return dispatch(searchExperimentsApi(params));
    },
    dispatchLoadMoreExperiemntsApi: (params) => {
      return dispatch(loadMoreExperimentsApi(params));
    },
    dispatchSearchInput: (input) => {
      return dispatch(experimentListSearchInput(input));
    },
  };
};

export default withRouter(
  connect(mapStateToProps, mapDispatchToProps)(WithDesignSystemThemeHoc(ExperimentListView)),
);
