import React from 'react';
import { css } from 'emotion';
import { SettingOutlined } from '@ant-design/icons';
import { Dropdown, Menu } from 'antd';
import { Button } from '../../shared/building_blocks/Button';
import { SearchTree } from '../../common/components/SearchTree';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { COLUMN_TYPES, ATTRIBUTE_COLUMN_LABELS } from '../constants';

import { FormattedMessage } from 'react-intl';

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
        <Menu.Item className={styles.menuItem} key='this-menu-needs-at-least-1-menu-item'>
          <SearchTree
            data={this.getData()}
            onCheck={this.handleCheck}
            checkedKeys={this.getCheckedKeys()}
            onSearchInputEscapeKeyPress={this.handleSearchInputEscapeKeyPress}
          />
        </Menu.Item>
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
          <SettingOutlined style={{ marginTop: 2 }} />
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

export function getCategorizedUncheckedKeys(checkedKeys, allKeys) {
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
    } else if (
      columnType !== COLUMN_TYPES.PARAMS &&
      columnType !== COLUMN_TYPES.METRICS &&
      columnType !== COLUMN_TYPES.TAGS
    ) {
      result[COLUMN_TYPES.ATTRIBUTES].push(columnType);
    }
  });
  return result;
}

const styles = {
  menuItem: css({
    '&:hover': {
      backgroundColor: 'inherit !important;',
    },
  }),
};
