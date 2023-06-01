/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 * This component consists of a checkbox tree select and a search input on top of which toggles and
 * highlight tree nodes while user type in a search prefix.
 */
import React from 'react';
import { Input, Tree, SearchIcon } from '@databricks/design-system';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { injectIntl } from 'react-intl';

export const NodeShape = {
  // display name of the node
  title: PropTypes.string.isRequired,
  // uniq key to identify this node
  key: PropTypes.string.isRequired,
  // an array of child nodes
  children: PropTypes.arrayOf(PropTypes.object),
};

type SearchTreeImplProps = {
  data?: any[]; // TODO: PropTypes.shape(NodeShape)
  onCheck?: (...args: any[]) => any;
  checkedKeys: any[];
  onSearchInputEscapeKeyPress: (...args: any[]) => any;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
};

type SearchTreeImplState = any;

export class SearchTreeImpl extends React.Component<SearchTreeImplProps, SearchTreeImplState> {
  state = {
    // Current expanded keys
    expandedKeys: [],
    // Current search input value
    searchValue: '',
    // Whether to automatically expand all parent node of current expanded keys
    autoExpandParent: true,
  };

  handleExpand = (expandedKeys: any) => {
    this.setState({
      expandedKeys,
      // we set autoExpandParent to false here because if it is true, we won't be able to toggle
      // COLLAPSE any subtree having an expanded child node.
      autoExpandParent: false,
    });
  };

  handleSearch = (e: any) => {
    const { value } = e.target;
    const { data } = this.props;
    const dataList = flattenDataToList(data);
    const expandedKeys = _.uniq(
      dataList
        .map((item) =>
          (item as any).title.toLowerCase().includes(value.toLowerCase())
            ? getParentKey((item as any).key, data)
            : null,
        )
        .filter((item) => !_.isEmpty(item)),
    );
    this.setState({
      expandedKeys,
      searchValue: value,
      autoExpandParent: true,
    });
  };

  handleSearchInputKeyUp = (e: any) => {
    const { onSearchInputEscapeKeyPress } = this.props;
    if (e.key === 'Escape' && onSearchInputEscapeKeyPress) {
      this.setState({ searchValue: '' });
      onSearchInputEscapeKeyPress();
    }
  };

  handleCheck = (checkedKeys: any) => {
    const { onCheck } = this.props;
    if (onCheck) {
      onCheck(checkedKeys, this.getAllKeys());
    }
  };

  getAllKeys() {
    const { data } = this.props;
    const dataList = flattenDataToList(data);
    return dataList.map((item) => (item as any).key);
  }

  renderTreeNodes = (data: any) => {
    const { searchValue } = this.state;
    return data.map((item: any) => {
      const index = item.title.toLowerCase().indexOf(searchValue.toLowerCase());
      const beforeStr = item.title.substring(0, index);
      const matchStr = item.title.substring(index, index + searchValue.length);
      const afterStr = item.title.substring(index + searchValue.length);
      const title =
        index > -1 ? (
          // We set the span title to display search tree node text on hover
          <span style={styles.treeNodeTextStyle} title={item.title}>
            {beforeStr}
            <span className='search-highlight' style={styles.searchHighlight}>
              {matchStr}
            </span>
            {afterStr}
          </span>
        ) : (
          // We set the span title to display search tree node text on hover
          <span style={styles.treeNodeTextStyle} title={item.title}>
            {item.title}
          </span>
        );
      if (item.children) {
        return {
          key: item.key,
          title,
          children: this.renderTreeNodes(item.children),
          'data-test-id': item.key,
          'data-testid': 'tree-node',
        };
      }
      return {
        key: item.key,
        title,
        'data-test-id': item.key,
        'data-testid': 'tree-node',
      };
    });
  };

  render() {
    const { data, checkedKeys, intl } = this.props;
    const { expandedKeys, autoExpandParent, searchValue } = this.state;
    return (
      <div>
        <Input
          css={{ marginBottom: 8 }}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search',
            description:
              // eslint-disable-next-line max-len
              'Placeholder text for input box to search for the columns names that could be selected or unselected to be rendered on the experiment runs table',
          })}
          value={searchValue}
          onChange={this.handleSearch}
          onKeyUp={this.handleSearchInputKeyUp}
          prefix={<SearchIcon />}
        />
        <Tree
          // @ts-expect-error TS(2322): Type '{ checkable: true; onCheck: (checkedKeys: an... Remove this comment to see the full error message
          checkable
          onCheck={this.handleCheck}
          onExpand={this.handleExpand}
          expandedKeys={expandedKeys}
          autoExpandParent={autoExpandParent}
          checkedKeys={checkedKeys}
          treeData={this.renderTreeNodes(data)}
        />
      </div>
    );
  }
}

/**
 * Flatten all nodes in `data `to `dataList`
 * @param {Array} data - tree shape data
 * @param {Array} dataList - an array as an out parameter to collect the flattened nodes
 * @returns {Array}
 */
export const flattenDataToList = (data: any, dataList = []) => {
  for (let i = 0; i < data.length; i++) {
    const node = data[i];
    const { key, title } = node;
    // @ts-expect-error TS(2322): Type 'any' is not assignable to type 'never'.
    dataList.push({ key, title });
    if (node.children) {
      flattenDataToList(node.children, dataList);
    }
  }
  return dataList;
};

/**
 * Given a node's key and entire tree data, find the key of its parent node
 * @param {string} key - key of the node to find parent for
 * @param {Array} treeData - entire tree data
 * @returns {string} parent key
 */
// @ts-expect-error TS(7023): 'getParentKey' implicitly has return type 'any' be... Remove this comment to see the full error message
export const getParentKey = (key: any, treeData: any) => {
  let parentKey;
  for (let i = 0; i < treeData.length; i++) {
    const node = treeData[i];
    if (node.children) {
      if (node.children.some((item: any) => item.key === key)) {
        parentKey = node.key;
      } else {
        parentKey = getParentKey(key, node.children);
      }
      if (parentKey) break;
    }
  }
  return parentKey;
};

export const styles = {
  treeNodeTextStyle: {
    maxWidth: 400,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  searchHighlight: { color: '#f50' },
};

// @ts-expect-error TS(2769): No overload matches this call.
export const SearchTree = injectIntl(SearchTreeImpl);
