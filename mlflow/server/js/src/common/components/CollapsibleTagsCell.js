import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../utils/Utils';
import { css } from 'emotion';

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
    const tagsToDisplay = this.state.collapsed
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
          return (
            <div className='tag-cell-item truncate-text single-line' key={tagName}>
              {value === '' ? (
                <span className='tag-name'>{tagName}</span>
              ) : (
                <span>
                  <span className='tag-name'>{tagName}:</span>
                  <span className='metric-param-value'>{value}</span>
                </span>
              )}
            </div>
          );
        })}
        {visibleTags.length > 3 ? (this.state.collapsed ? showMore : showLess) : null}
      </div>
    );
  }
}

const expandableListClassName = css({
  '.expander-text': {
    textDecoration: 'underline',
    cursor: 'pointer',
  },
});
