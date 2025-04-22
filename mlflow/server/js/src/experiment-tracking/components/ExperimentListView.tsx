import React, { Component } from 'react';
import { Theme } from '@emotion/react';
import {
  Checkbox,
  CaretDownSquareIcon,
  PlusCircleIcon,
  Input,
  PencilIcon,
  Typography,
  WithDesignSystemThemeHoc,
  DesignSystemHocProps,
  Button,
  TrashIcon,
  Tooltip,
} from '@databricks/design-system';
import { List as VList, AutoSizer, ListRowRenderer } from 'react-virtualized';
import 'react-virtualized/styles.css';
import { Link, NavigateFunction } from '../../common/utils/RoutingUtils';
import Routes from '../routes';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { withRouterNext, WithRouterNextProps } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';

type Props = {
  activeExperimentIds: string[];
  experiments: ExperimentEntity[];
} & WithRouterNextProps &
  DesignSystemHocProps;

type State = {
  checkedKeys: string[];
  hidden: boolean;
  searchInput: string;
  showCreateExperimentModal: boolean;
  showDeleteExperimentModal: boolean;
  showRenameExperimentModal: boolean;
  selectedExperimentId: string;
  selectedExperimentName: string;
};

export class ExperimentListView extends Component<Props, State> {
  list?: VList = undefined;

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

  bindListRef = (ref: VList) => {
    this.list = ref;
  };

  componentDidUpdate = () => {
    // Ensure the filter is applied
    if (this.list) {
      this.list.forceUpdateGrid();
    }
  };

  filterExperiments = (searchInput: string) => {
    const { experiments } = this.props;
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? this.props.experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  handleSearchInputChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    this.setState({
      searchInput: event.target.value,
    });
  };

  updateSelectedExperiment = (experimentId: string, experimentName: string) => {
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

  handleDeleteExperiment = (experimentId: string, experimentName: string) => () => {
    this.setState({
      showDeleteExperimentModal: true,
    });
    this.updateSelectedExperiment(experimentId, experimentName);
  };

  handleRenameExperiment = (experimentId: string, experimentName: string) => () => {
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
  handleCheck = (isChecked: boolean, key: string) => {
    this.setState((prevState, props) => {
      let { checkedKeys } = prevState;
      if (isChecked === true && !props.activeExperimentIds.includes(key)) {
        checkedKeys = [key, ...props.activeExperimentIds];
      }
      if (isChecked === false && props.activeExperimentIds.length !== 1) {
        checkedKeys = props.activeExperimentIds.filter((i) => i !== key);
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

  renderListItem: ListRowRenderer = ({ index, key, style, parent }) => {
    // Use the parents props to index.
    const item = parent.props['data'][index];
    const { activeExperimentIds } = this.props;
    const isActive = activeExperimentIds.includes(item.experimentId);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
    const { theme } = this.props.designSystemThemeApi;
    // Clicking the link removes all checks and marks other experiments
    // as not active.
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          // gap: theme.spacing.xs,
          paddingLeft: theme.spacing.xs,
          paddingRight: theme.spacing.xs,
          borderLeft: isActive ? `solid ${theme.colors.primary}` : 'solid transparent',
          borderLeftWidth: 4,
          backgroundColor: isActive ? theme.colors.actionDefaultBackgroundPress : 'transparent',
          fontSize: theme.typography.fontSizeBase,
          svg: {
            width: 14,
            height: 14,
          },
        }}
        data-testid={dataTestId}
        key={key}
        style={style}
      >
        <Checkbox
          componentId="mlflow.experiment_list_view.check_box"
          id={item.experimentId}
          key={item.experimentId}
          onChange={(isChecked) => this.handleCheck(isChecked, item.experimentId)}
          isChecked={isActive}
          data-testid={`${dataTestId}-check-box`}
        />
        <Link
          className="experiment-link"
          css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}
          to={Routes.getExperimentPageRoute(item.experimentId)}
          onClick={() => this.setState({ checkedKeys: [item.experimentId] })}
          title={item.name}
          data-testid={`${dataTestId}-link`}
        >
          {item.name}
        </Link>
        <Tooltip componentId="mlflow.experiment_list_view.rename_experiment_button.tooltip" content="Rename experiment">
          <Button
            type="link"
            componentId="mlflow.experiment_list_view.rename_experiment_button"
            icon={<PencilIcon />}
            onClick={this.handleRenameExperiment(item.experimentId, item.name)}
            data-testid="rename-experiment-button"
            size="small"
          />
        </Tooltip>
        <Tooltip componentId="mlflow.experiment_list_view.delete_experiment_button.tooltip" content="Delete experiment">
          <Button
            type="link"
            componentId="mlflow.experiment_list_view.delete_experiment_button"
            icon={<TrashIcon />}
            onClick={this.handleDeleteExperiment(item.experimentId, item.name)}
            data-testid="delete-experiment-button"
            size="small"
          />
        </Tooltip>
      </div>
    );
  };

  unHide = () => this.setState({ hidden: false });
  hide = () => this.setState({ hidden: true });

  render() {
    const { hidden } = this.state;
    const { activeExperimentIds, designSystemThemeApi } = this.props;
    const { theme } = designSystemThemeApi;

    if (hidden) {
      return (
        <Tooltip content="Show experiment list" componentId="mlflow.experiment_list_view.show_experiments.tooltip">
          <Button
            componentId="mlflow.experiment_list_view.show_experiments"
            icon={<CaretDownSquareIcon />}
            onClick={this.unHide}
            css={{ svg: { transform: 'rotate(-90deg)' } }}
            title="Show experiment list"
          />
        </Tooltip>
      );
    }

    const { searchInput } = this.state;
    const filteredExperiments = this.filterExperiments(searchInput);

    return (
      <div
        id="experiment-list-outer-container"
        css={{
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
        }}
      >
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
        <div
          css={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: theme.spacing.sm,
          }}
        >
          <Typography.Title level={2} style={{ margin: 0 }}>
            Experiments
          </Typography.Title>
          <div>
            <Tooltip componentId="mlflow.experiment_list_view.new_experiment_button.tooltip" content="New experiment">
              <Button
                componentId="mlflow.experiment_list_view.new_experiment_button"
                icon={<PlusCircleIcon />}
                onClick={this.handleCreateExperiment}
                title="New Experiment"
                data-testid="create-experiment-button"
              />
            </Tooltip>
            <Tooltip componentId="mlflow.experiment_list_view.hide_button.tooltip" content="Hide experiment list">
              <Button
                componentId="mlflow.experiment_list_view.hide_button"
                icon={<CaretDownSquareIcon />}
                onClick={this.hide}
                css={{ svg: { transform: 'rotate(90deg)' } }}
                title="Hide experiment list"
              />
            </Tooltip>
          </div>
        </div>
        <Input
          componentId="mlflow.experiment_list_view.search_input"
          placeholder="Search experiments"
          aria-label="search experiments"
          value={searchInput}
          onChange={this.handleSearchInputChange}
          data-testid="search-experiment-input"
        />
        <div css={{ marginTop: theme.spacing.xs }}>
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

export default withRouterNext(WithDesignSystemThemeHoc(ExperimentListView));
