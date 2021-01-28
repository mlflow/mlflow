import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../../common/utils/Utils';
import { Button } from 'antd';

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
    return (
      <div>
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
        {visibleTags.length > 3 ? (
          <Button type='link' className='tag-cell-toggle-link' onClick={this.handleToggleCollapse}>
            {this.state.collapsed
              ? `${visibleTags.length - NUM_TAGS_ON_COLLAPSED} more`
              : `Show less`}
          </Button>
        ) : null}
      </div>
    );
  }
}
