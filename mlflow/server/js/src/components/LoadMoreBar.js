/**
 "Load more" bar for user to click and load more runs. This row is currently built
 outside of the Table component as we are following a minimum-invasive way of building
 this feature to avoid massive refactor on current implementation. Ideally, this row
 can be built inside the Table as a special row by rewriting table rendering with a
 custom `rowRenderer`. That way, we don't need to handle scrolling position manually.
 We can consider doing this refactor while we implement the multi-level nested runs.
 TODO(Zangr) rewrite the runs table with rowRenderer to allow a built-in load-more row
*/

import React from 'react';
import { Button, Icon } from 'antd';
import PropTypes from 'prop-types';

export class LoadMoreBar extends React.PureComponent {
  static propTypes = {
    height: PropTypes.number.isRequired,
    width: PropTypes.number.isRequired,
    borderStyle: PropTypes.string.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    onLoadMore: PropTypes.func.isRequired,
  };

  render() {
    const { height, width, borderStyle, loadingMore, onLoadMore } = this.props;
    return (
      <div
        className='load-more-row'
        style={{ height, width, border: borderStyle }}
      >
        {loadingMore ? (
          <div className='loading-more-wrapper' style={{ display: 'flex', alignItems: 'center' }}>
            <Icon type='sync' spin style={{ fontSize: 20 }}/>
          </div>
        ) : (
          <Button
            className='load-more-button'
            type='primary'
            htmlType='button'
            onClick={onLoadMore}
            disabled={loadingMore}
            size='small'
          >
            Load more
          </Button>
        )}
      </div>
    );
  }
}
