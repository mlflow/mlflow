/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { useRef, useState, useEffect } from 'react';
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
import { ExperimentPageUIState } from './experiment-page/models/ExperimentPageUIState';
import { IconButton } from '../../common/components/IconButton';
import { withRouterNext } from '../../common/utils/withRouterNext';
import { ExperimentEntity } from '../types';

export type ExperimentListViewProps = {
  activeExperimentIds: string[];
  experiments: ExperimentEntity[];
  navigate: NavigateFunction;
  uiState: ExperimentPageUIState;
  setUIState: React.Dispatch<React.SetStateAction<ExperimentPageUIState>>;
} & DesignSystemHocProps;

export const ExperimentListView = ({
  activeExperimentIds,
  experiments,
  navigate,
  designSystemThemeApi,
  uiState,
  setUIState,
}: ExperimentListViewProps) => {
  const list = useRef(null);

  const [state, setState] = useState({
    checkedKeys: activeExperimentIds,
    searchInput: '',
    showCreateExperimentModal: false,
    showDeleteExperimentModal: false,
    showRenameExperimentModal: false,
    selectedExperimentId: '0',
    selectedExperimentName: '',
  });

  const filterExperiments = (searchInput: any) => {
    const lowerCasedSearchInput = searchInput.toLowerCase();
    return lowerCasedSearchInput === ''
      ? experiments
      : experiments.filter(({ name }) => name.toLowerCase().includes(lowerCasedSearchInput));
  };

  const handleSearchInputChange = (event: any) => {
    setState((state) => ({
      ...state,
      searchInput: event.target.value,
    }));
  };

  const updateSelectedExperiment = (experimentId: any, experimentName: any) => {
    setState((state) => ({
      ...state,
      selectedExperimentId: experimentId,
      selectedExperimentName: experimentName,
    }));
  };

  const handleCreateExperiment = () => {
    setState((state) => ({
      ...state,
      showCreateExperimentModal: true,
    }));
  };

  const handleDeleteExperiment = (experimentId: any, experimentName: any) => () => {
    setState((state) => ({
      ...state,
      showDeleteExperimentModal: true,
    }));
    updateSelectedExperiment(experimentId, experimentName);
  };

  const handleRenameExperiment = (experimentId: any, experimentName: any) => () => {
    setState((state) => ({
      ...state,
      showRenameExperimentModal: true,
    }));
    updateSelectedExperiment(experimentId, experimentName);
  };

  const handleCloseCreateExperimentModal = () => {
    setState((state) => ({
      ...state,
      showCreateExperimentModal: false,
    }));
  };

  const handleCloseDeleteExperimentModal = () => {
    setState((state) => ({
      ...state,
      showDeleteExperimentModal: false,
    }));
    // reset
    updateSelectedExperiment('0', '');
  };

  const handleCloseRenameExperimentModal = () => {
    setState((state) => ({
      ...state,
      showRenameExperimentModal: false,
    }));
    // reset
    updateSelectedExperiment('0', '');
  };

  useEffect(() => {
    if (state.checkedKeys.length > 0) {
      const route =
        state.checkedKeys.length === 1
          ? Routes.getExperimentPageRoute(state.checkedKeys[0])
          : Routes.getCompareExperimentsPageRoute(state.checkedKeys);
      navigate(route);
    }
  }, [state.checkedKeys, navigate]);

  // Add a key if it does not exist, remove it if it does
  // Always keep at least one experiment checked if it is only the active one.
  const handleCheck = (isChecked: any, key: any) => {
    setState((prevState) => {
      let { checkedKeys } = prevState;
      if (isChecked === true && !activeExperimentIds.includes(key)) {
        checkedKeys = [key, ...activeExperimentIds];
      }
      if (isChecked === false && activeExperimentIds.length !== 1) {
        checkedKeys = activeExperimentIds.filter((i: any) => i !== key);
      }
      return { ...prevState, checkedKeys: checkedKeys };
    });
  };

  // Avoid calling emotion for every list item
  const activeExperimentListItem = classNames.getExperimentListItemContainer(true, designSystemThemeApi.theme);
  const inactiveExperimentListItem = classNames.getExperimentListItemContainer(false, designSystemThemeApi.theme);

  const renderListItem = ({ index, key, style, isScrolling, parent }: any) => {
    // Use the parents props to index.
    const item = parent.props.data[index];
    const isActive = activeExperimentIds.includes(item.experimentId);
    const dataTestId = isActive ? 'active-experiment-list-item' : 'experiment-list-item';
    // Clicking the link removes all checks and marks other experiments
    // as not active.
    return (
      <div
        css={isActive ? activeExperimentListItem : inactiveExperimentListItem}
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
              onChange={(isChecked) => handleCheck(isChecked, item.experimentId)}
              isChecked={isActive}
              data-testid={`${dataTestId}-check-box`}
            ></Checkbox>,
            <Link
              className="experiment-link"
              to={Routes.getExperimentPageRoute(item.experimentId)}
              onClick={() => setState((state) => ({ ...state, checkedKeys: [item.experimentId] }))}
              title={item.name}
              data-testid={`${dataTestId}-link`}
            >
              {item.name}
            </Link>,
            <IconButton
              icon={<PencilIcon />}
              // @ts-expect-error TS(2322): Type '{ icon: Element; onClick: () => void; "data-... Remove this comment to see the full error message
              onClick={handleRenameExperiment(item.experimentId, item.name)}
              data-testid="rename-experiment-button"
              css={classNames.renameExperiment}
            />,
            <IconButton
              icon={<i className="far fa-trash-o" />}
              // @ts-expect-error TS(2322): Type '{ icon: Element; onClick: () => void; css: {... Remove this comment to see the full error message
              onClick={handleDeleteExperiment(item.experimentId, item.name)}
              css={classNames.deleteExperiment}
              data-testid="delete-experiment-button"
            />,
          ]}
        ></List.Item>
      </div>
    );
  };

  if (uiState.experimentListHidden) {
    return (
      <CaretDownSquareIcon
        rotate={-90}
        onClick={() => setUIState((uiState) => ({ ...uiState, experimentListHidden: false }))}
        css={classNames.icon(designSystemThemeApi.theme)}
        title="Show experiment list"
      />
    );
  }

  const filteredExperiments = filterExperiments(state.searchInput);

  return (
    <div id="experiment-list-outer-container" css={classNames.experimentListOuterContainer}>
      <CreateExperimentModal isOpen={state.showCreateExperimentModal} onClose={handleCloseCreateExperimentModal} />
      <DeleteExperimentModal
        isOpen={state.showDeleteExperimentModal}
        onClose={handleCloseDeleteExperimentModal}
        activeExperimentIds={activeExperimentIds}
        experimentId={state.selectedExperimentId}
        experimentName={state.selectedExperimentName}
      />
      <RenameExperimentModal
        isOpen={state.showRenameExperimentModal}
        onClose={handleCloseRenameExperimentModal}
        experimentId={state.selectedExperimentId}
        experimentName={state.selectedExperimentName}
      />
      <div css={classNames.experimentTitleContainer}>
        <Typography.Title level={2} style={{ margin: 0 }}>
          Experiments
        </Typography.Title>
        <div>
          <PlusCircleIcon
            onClick={handleCreateExperiment}
            css={classNames.icon(designSystemThemeApi.theme)}
            title="New Experiment"
            data-testid="create-experiment-button"
          />
          <CaretDownSquareIcon
            onClick={() => setUIState((uiState) => ({ ...uiState, experimentListHidden: true }))}
            rotate={90}
            css={classNames.icon(designSystemThemeApi.theme)}
            title="Hide experiment list"
          />
        </div>
      </div>
      <Input
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experimentlistview.tsx_280"
        placeholder="Search Experiments"
        aria-label="search experiments"
        value={state.searchInput}
        onChange={handleSearchInputChange}
        data-testid="search-experiment-input"
      />
      <div>
        <AutoSizer>
          {({ width, height }) => (
            <VList
              rowRenderer={renderListItem}
              data={filteredExperiments}
              ref={list}
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
};

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
