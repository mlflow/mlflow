/**
 * This component consists of a checkbox tree select and a search input on top of which toggles and
 * highlight tree nodes while user type in a search prefix.
 */
import React from 'react';
import { Input, Tree } from 'antd';
import _ from 'lodash';
import PropTypes from 'prop-types';

const { TreeNode } = Tree;
const { Search } = Input;
export const NodeShape = {
  // display name of the node
  title: PropTypes.string.isRequired,
  // uniq key to identify this node
  key: PropTypes.string.isRequired,
  // an array of child nodes
  children: PropTypes.arrayOf(PropTypes.object),
};

export class SearchTree extends React.Component {
  static propTypes = {
    // A forest of data nodes for rendering the checkbox tree view
    data: PropTypes.arrayOf(PropTypes.shape(NodeShape)),
    // Handler called when any node in the tree is checked/unchecked
    onCheck: PropTypes.func,
    // All keys checked
    checkedKeys: PropTypes.array.isRequired,
    // Handler called when user press ESC key in the search input
    onSearchInputEscapeKeyPress: PropTypes.func.isRequired,
  };

  state = {
    // Current expanded keys
    expandedKeys: [],
    // Current search input value
    searchValue: '',
    // Whether to automatically expand all parent node of current expanded keys
    autoExpandParent: true,
  };

  handleExpand = (expandedKeys) => {
    this.setState({
      expandedKeys,
      // we set autoExpandParent to false here because if it is true, we won't be able to toggle
      // COLLAPSE any subtree having an expanded child node.
      autoExpandParent: false,
    });
  };

  handleSearch = (e) => {
    const { value } = e.target;
    const { data } = this.props;
    const dataList = flattenDataToList(data);
    const expandedKeys = _.uniq(
      dataList
        .map((item) =>
          item.title.toLowerCase().includes(value.toLowerCase())
            ? getParentKey(item.key, data)
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

  handleSearchInputKeyUp = (e) => {
    const { onSearchInputEscapeKeyPress } = this.props;
    if (e.key === 'Escape' && onSearchInputEscapeKeyPress) {
      this.setState({ searchValue: '' });
      onSearchInputEscapeKeyPress();
    }
  };

  handleCheck = (checkedKeys) => {
    const { onCheck } = this.props;
    if (onCheck) {
      onCheck(checkedKeys, this.getAllKeys());
    }
  };

  getAllKeys() {
    const { data } = this.props;
    const dataList = flattenDataToList(data);
    return dataList.map((item) => item.key);
  }

  renderTreeNodes = (data) => {
    const { searchValue } = this.state;
    return data.map((item) => {
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
        return (
          <TreeNode data-test-id={item.key} key={item.key} title={title}>
            {this.renderTreeNodes(item.children)}
          </TreeNode>
        );
      }
      return <TreeNode data-test-id={item.key} key={item.key} title={title} />;
    });
  };

  render() {
    const { data, checkedKeys } = this.props;
    const { expandedKeys, autoExpandParent, searchValue } = this.state;
    return (
      <div>
        <Search
          style={{ marginBottom: 8 }}
          placeholder='Search'
          value={searchValue}
          onChange={this.handleSearch}
          onKeyUp={this.handleSearchInputKeyUp}
        />
        <Tree
          checkable
          onCheck={this.handleCheck}
          onExpand={this.handleExpand}
          expandedKeys={expandedKeys}
          autoExpandParent={autoExpandParent}
          checkedKeys={checkedKeys}
        >
          {this.renderTreeNodes(data)}
        </Tree>
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
export const flattenDataToList = (data, dataList = []) => {
  for (let i = 0; i < data.length; i++) {
    const node = data[i];
    const { key, title } = node;
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
export const getParentKey = (key, treeData) => {
  let parentKey;
  for (let i = 0; i < treeData.length; i++) {
    const node = treeData[i];
    if (node.children) {
      if (node.children.some((item) => item.key === key)) {
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
    display: 'inline-block',
    maxWidth: 400,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    marginRight: 20,
  },
  searchHighlight: { color: '#f50' },
};
