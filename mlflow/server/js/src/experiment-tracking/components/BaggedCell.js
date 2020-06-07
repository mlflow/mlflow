import React, { PureComponent } from 'react';
import PropTypes from 'prop-types';
import { Menu, Dropdown } from 'antd';
import classNames from 'classnames';
import ExperimentViewUtil from './ExperimentViewUtil';
import Utils from '../../common/utils/Utils';

const styles = {
  metricParamCellContent: {
    display: 'inline-block',
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

  handleSortAscending = () => {
    const { isParam, keyName, onSortBy } = this.props;
    const keyType = isParam ? 'params' : 'metrics';
    const canonicalKey = ExperimentViewUtil.makeCanonicalKey(keyType, keyName);
    onSortBy(canonicalKey, true);
  };

  handleSortDescending = () => {
    const { isParam, keyName, onSortBy } = this.props;
    const keyType = isParam ? 'params' : 'metrics';
    const canonicalKey = ExperimentViewUtil.makeCanonicalKey(keyType, keyName);
    onSortBy(canonicalKey, false);
  };

  handleRemoveBagged = () => {
    const { isParam, keyName, onRemoveBagged } = this.props;
    onRemoveBagged(isParam, keyName);
  };

  render() {
    const { isMetric, keyName, value, sortIcon } = this.props;
    const cellClass = classNames('metric-param-content', 'metric-param-cell', 'BaggedCell');
    return (
      <span className={cellClass}>
        <Dropdown
          overlay={
            <Menu>
              <Menu.Item data-test-id='sort-ascending' onClick={this.handleSortAscending}>
                Sort ascending
              </Menu.Item>
              <Menu.Item data-test-id='sort-descending' onClick={this.handleSortDescending}>
                Sort descending
              </Menu.Item>
              <Menu.Item data-test-id='remove-bagged' onClick={this.handleRemoveBagged}>
                Display as a separate column
              </Menu.Item>
            </Menu>
          }
          trigger={['click']}
        >
          <span>
            <span
              className='run-table-container underline-on-hover metric-param-sort-toggle'
              style={styles.metricParamCellContent}
              title={keyName}
            >
              {sortIcon}
              {keyName}:
            </span>
            <span
              className='metric-param-value run-table-container'
              style={styles.metricParamCellContent}
              title={value}
            >
              {isMetric ? Utils.formatMetric(parseFloat(value)) : value}
            </span>
          </span>
        </Dropdown>
      </span>
    );
  }
}
