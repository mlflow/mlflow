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
import { Button, Icon, Tooltip } from 'antd';
import PropTypes from 'prop-types';

export class LoadMoreBar extends React.PureComponent {
  static propTypes = {
    style: PropTypes.object,
    loadingMore: PropTypes.bool.isRequired,
    onLoadMore: PropTypes.func.isRequired,
    disableButton: PropTypes.bool,
    nestChildren: PropTypes.bool,
  };

  renderButton() {
    const { disableButton, onLoadMore, nestChildren } = this.props;
    const loadMoreButton = (
      <Button
        className='load-more-button'
        style={styles.loadMoreButton}
        type='primary'
        htmlType='button'
        onClick={onLoadMore}
        size='small'
        disabled={disableButton}
      >
        Load more
      </Button>
    );

    if (disableButton) {
      return (
        <Tooltip
          className='load-more-button-disabled-tooltip'
          placement='bottom'
          title='No more runs to load.'
        >
          {loadMoreButton}
        </Tooltip>
      );
    } else if (nestChildren) {
      return (
        <div>
          {loadMoreButton}
          <Tooltip
            className='load-more-button-nested-info-tooltip'
            placement='bottom'
            title='Loaded child runs are nested under their parents.'
          >
            <i className='fas fa-info-circle' style={styles.nestedTooltip} />
          </Tooltip>
        </div>
      );
    } else {
      return loadMoreButton;
    }
  }

  render() {
    const { loadingMore, style } = this.props;
    return (
      <div className='load-more-row' style={{ ...styles.loadMoreRows, ...style }}>
        {loadingMore ? (
          <div className='loading-more-wrapper' style={styles.loadingMoreWrapper}>
            <Icon type='sync' spin style={styles.loadingMoreIcon} />
          </div>
        ) : (
          this.renderButton()
        )}
      </div>
    );
  }
}

const styles = {
  loadMoreRows: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    background: 'white',
  },
  loadingMoreWrapper: {
    display: 'flex',
    alignItems: 'center',
  },
  loadingMoreIcon: {
    fontSize: 20,
  },
  loadMoreButton: {
    paddingLeft: 16,
    paddingRight: 16,
  },
  nestedTooltip: {
    color: '#2374BB', // matches antd primary button colour
    marginLeft: 8,
  },
};
