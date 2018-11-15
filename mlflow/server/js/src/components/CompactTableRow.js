import Utils from "../utils/Utils";
import React, { Component, PureComponent } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { Dropdown, MenuItem } from 'react-bootstrap';
import {RunInfo} from "../sdk/MlflowMessages";
import classNames from 'classnames';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';
import ExperimentViewUtil from "./ExperimentViewUtil";
import BaggedCell from "./BaggedCell";

const styles = {
  sortArrow: {
    marginLeft: "2px",
  },
  sortContainer: {
    minHeight: "18px",
  },
  sortToggle: {
    cursor: "pointer",
  },
  sortKeyName: {
    display: "inline-block"
  },
  metricParamCellContent: {
    display: "inline-block",
    maxWidth: 120,
  },
  metricParamNameContainer: {
    verticalAlign: "middle",
  },
};

export default class CompactTableRow extends Component {

  constructor(props) {
    super(props);
    this.onHover = this.onHover.bind(this);
  }

  state = {
    hoverState: {isMetric: false, isParam: false, key: ""},
  };

  static propTypes = {
    runInfo: PropTypes.object.isRequired,
    paramsMap: PropTypes.object.isRequired,
    metricsMap: PropTypes.object.isRequired,
    onCheckbox: PropTypes.func.isRequired,
    sortState: PropTypes.object.isRequired,
    runsSelected: PropTypes.object.isRequired,
    tagsMap: PropTypes.object.isRequired,
    setSortByHandler: PropTypes.func.isRequired,
    onExpand: PropTypes.func.isRequired,
    paramKeyList: PropTypes.arrayOf(String).isRequired,
    metricKeyList: PropTypes.arrayOf(String).isRequired,
    metricRanges: PropTypes.object.isRequired,
    unbaggedMetrics: PropTypes.arrayOf(String),
    unbaggedParams: PropTypes.arrayOf(String),
    onRemoveBagged: PropTypes.func.isRequired,
    isParent: PropTypes.bool.isRequired,
    hasExpander: PropTypes.bool.isRequired,
    expanderOpen: PropTypes.bool.isRequired,
    childrenIds: PropTypes.arrayOf(String),
  };


  onHover({isParam, isMetric, key}) {
    this.setState({ hoverState: {isParam, isMetric, key} });
  }


  shouldComponentUpdate(nextProps, nextState) {
    return nextState.hoverState !== this.state.hoverState;
    // return this.props.isHovered !== nextProps.isHovered;
  }

  /**
   * Returns true if our table should contain a column for displaying bagged params (if isParam is
   * truthy) or bagged metrics. TODO don't copy this
   */
  shouldShowBaggedColumn(isParam) {
    const { metricKeyList, paramKeyList, unbaggedMetrics, unbaggedParams } = this.props;
    if (isParam) {
      return unbaggedParams.length !== paramKeyList.length || paramKeyList.length === 0;
    }
    return unbaggedMetrics.length !== metricKeyList.length || metricKeyList.length === 0;
  }

  render() {
    const {
      runInfo,
      paramsMap,
      metricsMap,
      onCheckbox,
      sortState,
      runsSelected,
      tagsMap,
      setSortByHandler,
      onExpand,
      paramKeyList,
      metricKeyList,
      metricRanges,
      unbaggedMetrics,
      unbaggedParams,
      onRemoveBagged,
      isParent,
      hasExpander,
      expanderOpen,
      childrenIds
    } = this.props;
    const hoverState = this.state.hoverState;
    const selected = runsSelected[runInfo.run_uuid] === true;
    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid)),
      ExperimentViewUtil.getExpander(
        hasExpander, expanderOpen, () => onExpand(runInfo.run_uuid, childrenIds), runInfo.run_uuid),
    ];
    ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsMap, isParent)
      .forEach((col) => rowContents.push(col));

    const unbaggedParamSet = new Set(unbaggedParams);
    const unbaggedMetricSet = new Set(unbaggedMetrics);
    const baggedParams = paramKeyList.filter((paramKey) =>
      !unbaggedParamSet.has(paramKey) && paramsMap[paramKey] !== undefined);
    const baggedMetrics = metricKeyList.filter((metricKey) =>
      !unbaggedMetricSet.has(metricKey) && metricsMap[metricKey] !== undefined);

    // Add params (unbagged, then bagged)
    unbaggedParams.forEach((paramKey) => {
      rowContents.push(ExperimentViewUtil.getUnbaggedParamCell(paramKey, paramsMap));
    });
    // Add bagged params
    const paramsCellContents = baggedParams.map((paramKey) => {
      const isHovered = hoverState.isParam && hoverState.key === paramKey;
      const keyname = "param-" + paramKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, false, true, paramKey);
      return (<BaggedCell
        key={keyname}
        keyName={paramKey} value={paramsMap[paramKey].getValue()} onHover={this.onHover}
        setSortByHandler={setSortByHandler} isMetric={false} isParam={true} isHovered={isHovered}
        onRemoveBagged={onRemoveBagged}/>);
    });
    if (this.shouldShowBaggedColumn(true)) {
      rowContents.push(
        <td key={"params-container-cell-" + runInfo.run_uuid} className="left-border">
          <div>{paramsCellContents}</div>
        </td>);
    }

    // Add metrics (unbagged, then bagged)
    unbaggedMetrics.forEach((metricKey) => {
      rowContents.push(
        ExperimentViewUtil.getUnbaggedMetricCell(metricKey, metricsMap, metricRanges));
    });

    // Add bagged metrics
    const metricsCellContents = baggedMetrics.map((metricKey) => {
      const keyname = "metric-" + metricKey;
      const isHovered = hoverState.isMetric && hoverState.key === metricKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, true, false, metricKey);
      const metric = metricsMap[metricKey].getValue();
      return (
        // TODO sorticon in bagged cells
        <BaggedCell key={keyname}
                    keyName={metricKey} value={Utils.formatMetric(metric)} onHover={this.onHover}
                    setSortByHandler={setSortByHandler} isMetric={true} isParam={false} isHovered={isHovered}
                    onRemoveBagged={onRemoveBagged}/>
      );
    });
    if (this.shouldShowBaggedColumn(false)) {
      rowContents.push(
        <td key={"metrics-container-cell-" + runInfo.run_uuid} className="metric-param-container-cell left-border">
          {metricsCellContents}
        </td>
      );
    }
    return rowContents;
  }
}