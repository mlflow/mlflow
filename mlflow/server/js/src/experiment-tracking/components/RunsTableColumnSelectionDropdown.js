import React from 'react';
import { Dropdown, Icon, Menu } from 'antd';
import { Button } from '../../shared/building_blocks/Button';
import { Checkbox } from '../../shared/building_blocks/Checkbox';
import { SearchTree } from '../../common/components/SearchTree';
import Utils from '../../common/utils/Utils';
import ExperimentViewUtil from './ExperimentViewUtil';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { COLUMN_TYPES, ATTRIBUTE_COLUMN_LABELS } from '../constants';

import { FormattedMessage } from 'react-intl';

export class RunsTableColumnSelectionDropdown extends React.Component {
  static propTypes = {
    paramKeyList: PropTypes.array,
    metricKeyList: PropTypes.array,
    visibleTagKeyList: PropTypes.array,
    runInfos: PropTypes.array,
    paramsList: PropTypes.array,
    metricsList: PropTypes.array,
    tagsList: PropTypes.array,
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
    diffViewSelected: false,
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
      ...Object.values(ATTRIBUTE_COLUMN_LABELS).map((text) => ({
        key: `${COLUMN_TYPES.ATTRIBUTES}-${text}`,
        title: text,
      })),
    ];

    // Parameters
    if (paramKeyList.length > 0) {
      data.push({
        title: 'Parameters',
        key: COLUMN_TYPES.PARAMS,
        children: paramKeyList.map((key) => ({ key: `${COLUMN_TYPES.PARAMS}-${key}`, title: key })),
      });
    }

    // Metrics
    if (metricKeyList.length > 0) {
      data.push({
        title: 'Metrics',
        key: COLUMN_TYPES.METRICS,
        children: metricKeyList.map((key) => ({
          key: `${COLUMN_TYPES.METRICS}-${key}`,
          title: key,
        })),
      });
    }

    // Tags
    if (visibleTagKeyList.length > 0) {
      data.push({
        title: 'Tags',
        key: COLUMN_TYPES.TAGS,
        children: visibleTagKeyList.map((key) => ({
          key: `${COLUMN_TYPES.TAGS}-${key}`,
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
        Object.values(ATTRIBUTE_COLUMN_LABELS),
        categorizedUncheckedKeys[COLUMN_TYPES.ATTRIBUTES],
      ).map((key) => `${COLUMN_TYPES.ATTRIBUTES}-${key}`),
      ..._.difference(paramKeyList, categorizedUncheckedKeys[COLUMN_TYPES.PARAMS]).map(
        (key) => `${COLUMN_TYPES.PARAMS}-${key}`,
      ),
      ..._.difference(metricKeyList, categorizedUncheckedKeys[COLUMN_TYPES.METRICS]).map(
        (key) => `${COLUMN_TYPES.METRICS}-${key}`,
      ),
      ..._.difference(visibleTagKeyList, categorizedUncheckedKeys[COLUMN_TYPES.TAGS]).map(
        (key) => `${COLUMN_TYPES.TAGS}-${key}`,
      ),
    ];
  }

  getCategorizedColumnsDiffView() {
    const {
      paramKeyList,
      metricKeyList,
      visibleTagKeyList,
      runInfos,
      paramsList,
      metricsList,
      tagsList,
    } = this.props;
    const attributeKeyList = [
      ATTRIBUTE_COLUMN_LABELS.RUN_NAME,
      ATTRIBUTE_COLUMN_LABELS.USER,
      ATTRIBUTE_COLUMN_LABELS.SOURCE,
      ATTRIBUTE_COLUMN_LABELS.VERSION,
    ];
    let attributes = [];
    let params = [];
    let metrics = [];
    let tags = [];

    for (let index = 0, n = runInfos.length; index < n; ++index) {
      const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[index]);
      const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[index]);
      const tagsMap = tagsList[index];

      attributes.push([
        Utils.getRunName(tagsList[index]),
        Utils.getUser(runInfos[index], tagsList[index]),
        Utils.formatSource(tagsList[index]),
        Utils.getSourceVersion(tagsList[index]),
      ]);
      params.push(
        paramKeyList.map((paramKey) => {
          return paramsMap[paramKey] ? paramsMap[paramKey].getValue() : '';
        }),
      );
      metrics.push(
        metricKeyList.map((metricKey) => {
          return metricsMap[metricKey] ? metricsMap[metricKey].getValue() : '';
        }),
      );
      tags.push(
        visibleTagKeyList.map((tagKey) => {
          return tagsMap[tagKey] ? tagsMap[tagKey].getValue() : '';
        }),
      );
    }
    // Transpose the matrices so that we can evaluate the values 'column-based'
    attributes = _.unzip(attributes);
    params = _.unzip(params);
    metrics = _.unzip(metrics);
    tags = _.unzip(tags);
    const allEqual = (arr) => arr.every((val) => val === arr[0]);

    return {
      [COLUMN_TYPES.ATTRIBUTES]: attributeKeyList.filter((v, index) => {
        return allEqual(attributes[index]);
      }),
      [COLUMN_TYPES.PARAMS]: paramKeyList.filter((v, index) => {
        return allEqual(params[index]);
      }),
      [COLUMN_TYPES.METRICS]: metricKeyList.filter((v, index) => {
        return allEqual(metrics[index]);
      }),
      [COLUMN_TYPES.TAGS]: visibleTagKeyList.filter((v, index) => {
        return allEqual(tags[index]);
      }),
    };
  }

  handleDiffViewCheckboxChange = () => {
    const { onCheck } = this.props;
    this.setState({ diffViewSelected: !this.state.diffViewSelected }, () => {
      const categorizedUncheckedKeys = this.state.diffViewSelected
        ? this.getCategorizedColumnsDiffView()
        : {
            [COLUMN_TYPES.ATTRIBUTES]: [],
            [COLUMN_TYPES.PARAMS]: [],
            [COLUMN_TYPES.METRICS]: [],
            [COLUMN_TYPES.TAGS]: [],
          };
      onCheck(categorizedUncheckedKeys);
    });
  };

  handleSearchInputEscapeKeyPress = () => {
    this.setState({ menuVisible: false });
  };

  handleVisibleChange = (menuVisible) => {
    this.setState({ menuVisible });
  };

  render() {
    const { menuVisible, diffViewSelected } = this.state;
    const content = (
      <Menu style={{ maxHeight: 480, overflowY: 'scroll' }}>
        <Checkbox
          label='Diff View'
          isSelected={diffViewSelected}
          onCheckboxChange={this.handleDiffViewCheckboxChange}
        />
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
      >
        <Button
          style={{ display: 'flex', alignItems: 'center' }}
          dataTestId='column-selection-dropdown'
        >
          <Icon type='setting' style={{ marginTop: 2 }} />
          <FormattedMessage
            defaultMessage='Columns'
            // eslint-disable-next-line max-len
            description='Dropdown text to display columns names that could to be rendered for the experiment runs table'
          />
        </Button>
      </Dropdown>
    );
  }
}

function getCategorizedUncheckedKeys(checkedKeys, allKeys) {
  const uncheckedKeys = _.difference(allKeys, checkedKeys);
  const result = {
    [COLUMN_TYPES.ATTRIBUTES]: [],
    [COLUMN_TYPES.PARAMS]: [],
    [COLUMN_TYPES.METRICS]: [],
    [COLUMN_TYPES.TAGS]: [],
  };
  uncheckedKeys.forEach((key) => {
    // split on first instance of '-' in case there are keys with '-'
    const [columnType, rawKey] = key.split(/-(.+)/);
    if (rawKey) {
      result[columnType].push(rawKey);
    } else if (columnType !== COLUMN_TYPES.PARAMS && columnType !== COLUMN_TYPES.METRICS) {
      result[COLUMN_TYPES.ATTRIBUTES].push(columnType);
    }
  });
  return result;
}
