import React from 'react';
import { Button, Dropdown, Icon, Menu } from 'antd';
import { SearchTree } from '../../common/components/SearchTree';
import PropTypes from 'prop-types';
import _ from 'lodash';
import ExperimentViewUtil from './ExperimentViewUtil';
import { ColumnTypes } from '../constants';

export class RunsTableColumnSelectionDropdown extends React.Component {
  static propTypes = {
    paramKeyList: PropTypes.array,
    metricKeyList: PropTypes.array,
    visibleTagKeyList: PropTypes.array,
    onCheck: PropTypes.func,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
  };

  static defaultProps = {
    paramKeyList: [],
    metricKeyList: [],
    visibleTagKeyList: [],
  };

  state = {
    menuVisible: false,
  };

  handleCheck = (checkedKeys, allKeys) => {
    const { onCheck } = this.props;
    if (onCheck) {
      onCheck(getCategorizedUncheckedKeys(checkedKeys, allKeys));
    }
  };

  getData() {
    const { paramKeyList, metricKeyList, visibleTagKeyList } = this.props;

    // Attributes
    const data = [
      ...Object.values(ExperimentViewUtil.AttributeColumnLabels).map((text) => ({
        key: `${ColumnTypes.ATTRIBUTES}-${text}`,
        title: text,
      })),
    ];

    // Parameters
    if (paramKeyList.length > 0) {
      data.push({
        title: 'Parameters',
        key: ColumnTypes.PARAMS,
        children: paramKeyList.map((key) => ({ key: `${ColumnTypes.PARAMS}-${key}`, title: key })),
      });
    }

    // Metrics
    if (metricKeyList.length > 0) {
      data.push({
        title: 'Metrics',
        key: ColumnTypes.METRICS,
        children: metricKeyList.map((key) => ({
          key: `${ColumnTypes.METRICS}-${key}`,
          title: key,
        })),
      });
    }

    // Tags
    if (visibleTagKeyList.length > 0) {
      data.push({
        title: 'Tags',
        key: ColumnTypes.TAGS,
        children: visibleTagKeyList.map((key) => ({
          key: `${ColumnTypes.TAGS}-${key}`,
          title: key,
        })),
      });
    }

    return data;
  }

  getCheckedKeys() {
    const { paramKeyList, metricKeyList, visibleTagKeyList, categorizedUncheckedKeys } = this.props;
    return [
      ..._.difference(
        Object.values(ExperimentViewUtil.AttributeColumnLabels),
        categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES],
      ).map((key) => `${ColumnTypes.ATTRIBUTES}-${key}`),
      ..._.difference(paramKeyList, categorizedUncheckedKeys[ColumnTypes.PARAMS]).map(
        (key) => `${ColumnTypes.PARAMS}-${key}`,
      ),
      ..._.difference(metricKeyList, categorizedUncheckedKeys[ColumnTypes.METRICS]).map(
        (key) => `${ColumnTypes.METRICS}-${key}`,
      ),
      ..._.difference(visibleTagKeyList, categorizedUncheckedKeys[ColumnTypes.TAGS]).map(
        (key) => `${ColumnTypes.TAGS}-${key}`,
      ),
    ];
  }

  handleSearchInputEscapeKeyPress = () => {
    this.setState({ menuVisible: false });
  };

  handleVisibleChange = (menuVisible) => {
    this.setState({ menuVisible });
  };

  render() {
    const { menuVisible } = this.state;
    const content = (
      <Menu style={{ maxHeight: 480, overflowY: 'scroll' }}>
        <SearchTree
          data={this.getData()}
          onCheck={this.handleCheck}
          checkedKeys={this.getCheckedKeys()}
          onSearchInputEscapeKeyPress={this.handleSearchInputEscapeKeyPress}
        />
      </Menu>
    );
    return (
      <Dropdown
        overlay={content}
        trigger={['click']}
        visible={menuVisible}
        onVisibleChange={this.handleVisibleChange}
        className='column-selection-dropdown'
      >
        <Button style={{ height: 34, display: 'flex', alignItems: 'center' }}>
          <Icon type='setting' style={{ marginTop: 2 }} />
          Columns
        </Button>
      </Dropdown>
    );
  }
}

function getCategorizedUncheckedKeys(checkedKeys, allKeys) {
  const uncheckedKeys = _.difference(allKeys, checkedKeys);
  const result = {
    [ColumnTypes.ATTRIBUTES]: [],
    [ColumnTypes.PARAMS]: [],
    [ColumnTypes.METRICS]: [],
    [ColumnTypes.TAGS]: [],
  };
  uncheckedKeys.forEach((key) => {
    // split on first instance of '-' in case there are keys with '-'
    const [columnType, rawKey] = key.split(/-(.+)/);
    if (rawKey) {
      result[columnType].push(rawKey);
    } else if (columnType !== ColumnTypes.PARAMS && columnType !== ColumnTypes.METRICS) {
      result[ColumnTypes.ATTRIBUTES].push(columnType);
    }
  });
  return result;
}
