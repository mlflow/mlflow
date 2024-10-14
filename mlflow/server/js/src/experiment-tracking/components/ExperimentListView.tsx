/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { css, Theme } from '@emotion/react';
import {
  Checkbox,
  CaretDownSquareIcon,
  PlusCircleIcon,
  Input,
  PencilIcon,
  Typography,
  WithDesignSystemThemeHoc,
  DesignSystemHocProps,
} from '@databricks/design-system';
import { List } from 'antd';
import { List as VList, AutoSizer } from 'react-virtualized';
import 'react-virtualized/styles.css';
import { Link, NavigateFunction } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { IconButton } from '../../common/components/IconButton';
import { withRouterNext } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';

type Props = {
  activeExperimentIds: string[];
  experiments: ExperimentEntity[];
  navigate: NavigateFunction;
} & DesignSystemHocProps;

type State = any;

export class ExperimentListView extends Component<Props, State> {
  list: any;

  state = {
    checkedKeys: this.props.activeExperimentIds,
    hidden: false,
    searchInput: '',
    showCreateExperimentModal: false,
    showDeleteExperimentModal: false,
    showRenameExperimentModal: false,
    selectedExperimentId: '0',
    selectedExperimentName: '',
  };

  bindListRef = (ref: any) => {
    this.list = ref;
  };

  componentDidUpdate = () => {
    // Ensure the filter is applied
    if (this.list) {
      this.list.forceUpdateGrid();
    }
  };

  filterExperiments = (searchInput: any) => {
    const { experiments } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? this.props.experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  handleSearchInputChange = (event: any) => {
    this.setState({
      searchInput: event.target.value,
    });
  };

  updateSelectedExperiment = (experimentId: any, experimentName: any) => {
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

  handleDeleteExperiment = (experimentId: any, experimentName: any) => () => {
    this.setState({
      showDeleteExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleRenameExperiment = (experimentId: any, experimentName: any) => () => {
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

  // Add a key if it does not exist, remove it if it does
  // Always keep at least one experiment checked if it is only the active one.
  handleCheck = (isChecked: any, key: any) => {
    this.setState((prevState: any, props: any) => {
      let { checkedKeys } = prevState;
      if (isChecked === true && !props.activeExperimentIds.includes(key)) {
        checkedKeys = [key, ...props.activeExperimentIds];
      }
      if (isChecked === false && props.activeExperimentIds.length !== 1) {
        checkedKeys = props.activeExperimentIds.filter((i: any) => i !== key);
      }
      return { checkedKeys: checkedKeys };
    }, this.pushExperimentRoute);
  };

  pushExperimentRoute = () => {
    if (this.state.checkedKeys.length > 0) {
      const route =
        this.state.checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(this.state.checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(this.state.checkedKeys);
      this.props.navigate(route);
    }
  };

  // Avoid calling emotion for every list item
  activeExperimentListItem = classNames.getExperimentListItemContainer(true, this.props.designSystemThemeApi.theme);
  inactiveExperimentListItem = classNames.getExperimentListItemContainer(false, this.props.designSystemThemeApi.theme);

  renderListItem = ({ index, key, style, isScrolling, parent }: any) => {
    // Use the parents props to index.
    const item = parent.props.data[index];
    const { activeExperimentIds } = this.props;
    const isActive = activeExperimentIds.includes(item.experimentId);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
    // Clicking the link removes all checks and marks other experiments
    // as not active.
    return (
      <div
        css={isActive ? this.activeExperimentListItem : this.inactiveExperimentListItem}
        data-testid={dataTestId}
        key={key}
        style={style}
      >
        <List.Item
          key={item.experimentId}
          // @ts-expect-error TS(2322): Type '{ key: any; bordered: string; prefixCls: str... Remove this comment to see the full error message
          bordered="false"
          prefixCls="experiment-list-meta"
          css={classNames.experimentListItem}
          actions={[
            <Checkbox
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experimentlistview.tsx_180"
              id={item.experimentId}
              key={item.experimentId}
              onChange={(isChecked) => this.handleCheck(isChecked, item.experimentId)}
              isChecked={isActive}
              data-testid={`${dataTestId}-check-box`}
            ></Checkbox>,
            <Link
              className="experiment-link"
              to={Routes.getExperimentPageRoute(item.experimentId)}
              onClick={() => this.setState({ checkedKeys: [item.experimentId] })}
              title={item.name}
              data-testid={`${dataTestId}-link`}
            >
              {item.name}
            </Link>,
            <IconButton
              icon={<PencilIcon />}
              // @ts-expect-error TS(2322): Type '{ icon: Element; onClick: () => void; "data-... Remove this comment to see the full error message
              onClick={this.handleRenameExperiment(item.experimentId, item.name)}
              data-testid="rename-experiment-button"
              css={classNames.renameExperiment}
            />,
            <IconButton
              icon={<i className="far fa-trash-o" />}
              // @ts-expect-error TS(2322): Type '{ icon: Element; onClick: () => void; css: {... Remove this comment to see the full error message
              onClick={this.handleDeleteExperiment(item.experimentId, item.name)}
              css={classNames.deleteExperiment}
              data-testid="delete-experiment-button"
            />,
          ]}
        ></List.Item>
      </div>
    );
  };

  isRowLoaded = ({ index }: any) => {
    return !!this.props.experiments[index];
  };

  unHide = () => this.setState({ hidden: false });
  hide = () => this.setState({ hidden: true });

  render() {
    const { hidden } = this.state;
    const { activeExperimentIds, designSystemThemeApi } = this.props;
    const { theme } = designSystemThemeApi;

    if (hidden) {
      return (
        <CaretDownSquareIcon
          rotate={-90}
          onClick={this.unHide}
          css={classNames.icon(theme)}
          title="Show experiment list"
        />
      );
    }

    const { searchInput } = this.state;
    const filteredExperiments = this.filterExperiments(searchInput);

    return (
      <div id="experiment-list-outer-container" css={classNames.experimentListOuterContainer}>
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
          <Typography.Title level={2} style={{ margin: 0 }}>
            Experiments
          </Typography.Title>
          <div>
            <PlusCircleIcon
              onClick={this.handleCreateExperiment}
              css={classNames.icon(theme)}
              title="New Experiment"
              data-testid="create-experiment-button"
            />
            <CaretDownSquareIcon
              onClick={this.hide}
              rotate={90}
              css={classNames.icon(theme)}
              title="Hide experiment list"
            />
          </div>
        </div>
        <Input
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experimentlistview.tsx_280"
          placeholder="Search Experiments"
          aria-label="search experiments"
          value={searchInput}
          onChange={this.handleSearchInputChange}
          data-testid="search-experiment-input"
        />
        <div>
          <AutoSizer>
            {({ width, height }) => (
              <VList
                rowRenderer={this.renderListItem}
                data={filteredExperiments}
                ref={this.bindListRef}
                rowHeight={32}
                overscanRowCount={10}
                height={height}
                width={width}
                rowCount={filteredExperiments.length}
              />
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
    height: '100%',
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
  getExperimentListItemContainer: (isActive: any, theme: any) =>
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
  icon: (theme: Theme) => ({
    color: theme.colors.actionDefaultTextDefault,
    fontSize: theme.general.iconSize,
    marginLeft: theme.spacing.xs,
  }),
};

export default withRouterNext(WithDesignSystemThemeHoc(ExperimentListView));
