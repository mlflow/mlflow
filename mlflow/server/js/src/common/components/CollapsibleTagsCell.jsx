import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../utils/Utils';
import { css } from 'emotion';
import { Tooltip } from '@databricks/design-system';

// Number of tags shown when cell is collapsed
export const NUM_TAGS_ON_COLLAPSED = 3;

export class CollapsibleTagsCell extends React.Component {
  static propTypes = {
    tags: PropTypes.object.isRequired,
    onToggle: PropTypes.func,
  };

  state = {
    collapsed: true,
  };

  handleToggleCollapse = () => {
    this.setState((prevState) => ({ collapsed: !prevState.collapsed }));
    if (this.props.onToggle) {
      this.props.onToggle();
    }
  };

  render() {
    const visibleTags = Utils.getVisibleTagValues(this.props.tags);
    // In cases where only 1 tag would be shown on expansion,
    // we should just show the tag instead of the expansion button
    const showCellExpansion = visibleTags.length > NUM_TAGS_ON_COLLAPSED + 1;
    const tagsToDisplay =
      this.state.collapsed && showCellExpansion
        ? visibleTags.slice(0, NUM_TAGS_ON_COLLAPSED)
        : visibleTags;
    const showLess = (
      <div onClick={this.handleToggleCollapse} className='expander-text'>
        Less
      </div>
    );
    const showMore = (
      <div onClick={this.handleToggleCollapse} className='expander-text'>
        +{visibleTags.length - NUM_TAGS_ON_COLLAPSED} more
      </div>
    );
    return (
      <div className={expandableListClassName}>
        {tagsToDisplay.map((entry) => {
          const tagName = entry[0];
          const value = entry[1];
          const tooltipContent = value === '' ? tagName : `${tagName}: ${value}`;
          return (
            <div className='tag-cell-item truncate-text single-line' key={tagName}>
                <Tooltip title={tooltipContent} placement="bottom">
                  <span className={overflowWithEllipsis}>
                    {value === '' ? (
                      <span className='tag-name'>{tagName}</span>
                    ) : (
                      <span>
                        <span className='tag-name'>{tagName}:</span>
                        <span className='metric-param-value'>{value}</span>
                      </span>
                    )}
                  </span>
                </Tooltip>
            </div>
          );
        })}
        {showCellExpansion ? (this.state.collapsed ? showMore : showLess) : null}
      </div>
    );
  }
}

const expandableListClassName = css({
  '.expander-text': {
    textDecoration: 'underline',
    cursor: 'pointer',
  },
  '.tag-cell-item': {
    display: 'flex',
  }
});

const overflowWithEllipsis = css({
  textOverflow: 'ellipsis',
  overflow: 'hidden',
  flexShrink: 1,
});
