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
import { injectIntl, FormattedMessage } from 'react-intl';

export class LoadMoreBarImpl extends React.PureComponent {
  static propTypes = {
    style: PropTypes.object,
    loadingMore: PropTypes.bool.isRequired,
    onLoadMore: PropTypes.func.isRequired,
    disableButton: PropTypes.bool,
    nestChildren: PropTypes.bool,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  renderButton() {
    const { disableButton, onLoadMore, nestChildren, intl } = this.props;
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
        <FormattedMessage
          defaultMessage='Load more'
          description='Load more button text to load more experiment runs'
        />
      </Button>
    );

    if (disableButton) {
      return (
        <Tooltip
          className='load-more-button-disabled-tooltip'
          placement='bottom'
          title={intl.formatMessage({
            defaultMessage: 'No more runs to load.',
            description:
              'Tooltip text for load more button when there are no more experiment runs to load',
          })}
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
            title={intl.formatMessage({
              defaultMessage: 'Loaded child runs are nested under their parents.',
              description:
                // eslint-disable-next-line max-len
                'Tooltip text for load more button explaining the runs are nested under their parent experiment run',
            })}
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

export const LoadMoreBar = injectIntl(LoadMoreBarImpl);
