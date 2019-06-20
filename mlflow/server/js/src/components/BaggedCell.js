import React, { PureComponent } from 'react';
import PropTypes from 'prop-types';
import { Dropdown, MenuItem } from 'react-bootstrap';
import classNames from 'classnames';
import ExperimentViewUtil from "./ExperimentViewUtil";
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';
import EmptyIfClosedMenu from './EmptyIfClosedMenu';

const styles = {
  metricParamCellContent: {
    display: "inline-block",
    maxWidth: 120,
  },
};

export default class BaggedCell extends PureComponent {
  static propTypes = {
    keyName: PropTypes.string.isRequired,
    value: PropTypes.string.isRequired,
    onSortBy: PropTypes.func.isRequired,
    isParam: PropTypes.bool.isRequired,
    isMetric: PropTypes.bool.isRequired,
    onRemoveBagged: PropTypes.func.isRequired,
    sortIcon: PropTypes.node,
  };

  render() {
    const { keyName, value, onSortBy, isParam, onRemoveBagged, sortIcon } = this.props;
    const keyType = (isParam ? "params" : "metrics");
    const canonicalKey = ExperimentViewUtil.makeCanonicalKey(keyType, keyName);
    const cellClass = classNames("metric-param-content", "metric-param-cell", "BaggedCell");
    return (
      <span
        className={cellClass}
      >
      <Dropdown id="dropdown-custom-1" style={{width: 250}}>
        <ExperimentRunsSortToggle
          bsRole="toggle"
          className={"metric-param-sort-toggle"}
        >
              <span
                className="run-table-container underline-on-hover"
                style={styles.metricParamCellContent}
                title={keyName}
              >
                {sortIcon}
                {keyName}:
              </span>
        </ExperimentRunsSortToggle>
        <span
          className="metric-param-value run-table-container"
          style={styles.metricParamCellContent}
        >
              {value}
        </span>
        <EmptyIfClosedMenu className="mlflow-menu" bsRole="menu">
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => onSortBy(canonicalKey, true)}
          >
            Sort ascending
          </MenuItem>
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => onSortBy(canonicalKey, false)}
          >
            Sort descending
          </MenuItem>
          <MenuItem
            className="mlflow-menu-item"
            onClick={() => onRemoveBagged(isParam, keyName)}
          >
            Display in own column
          </MenuItem>
        </EmptyIfClosedMenu>
      </Dropdown>
      </span>
    );
  }
}
