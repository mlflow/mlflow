import React, { PureComponent } from 'react';
import PropTypes from 'prop-types';
import { Menu, Dropdown } from 'antd';
import classNames from 'classnames';
import ExperimentViewUtil from "./ExperimentViewUtil";
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';

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
        <Dropdown
          overlay={(
            <Menu>
              <Menu.Item onClick={() => onSortBy(canonicalKey, true)}>
                Sort ascending
              </Menu.Item>
              <Menu.Item onClick={() => onSortBy(canonicalKey, false)}>
                Sort descending
              </Menu.Item>
              <Menu.Item onClick={() => onRemoveBagged(isParam, keyName)}>
                Display in own column
              </Menu.Item>
            </Menu>
          )}
          trigger='click'
        >
          <span>
            <ExperimentRunsSortToggle
                bsRole="toggle"
                className={"metric-param-sort-toggle"}
              >
              <span
                className="run-table-container underline-on-hover metric-param-sort-toggle"
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
          </span>
        </Dropdown>
      </span>
    );
  }
}

