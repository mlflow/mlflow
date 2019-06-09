import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../utils/Utils';

const NUM_VISIBLE_TAGS_WHEN_COLLAPSED = 3;

export class CollapsibleTagsCell extends React.Component {
  static propTypes = {
    tags: PropTypes.object.isRequired,
    onToggle: PropTypes.func,
  };

  state = {
    collapsed: true,
  };

  handleToggleCollapse = () => {
    const { onToggle } = this.props;
    this.setState((prevState) => ({ collapsed: !prevState.collapsed }));
    onToggle && onToggle();
  };

  render() {
    const visibleTags = Utils.getVisibleTagValues(this.props.tags);
    const tagsToDisplay = this.state.collapsed
      ? visibleTags.slice(0, NUM_VISIBLE_TAGS_WHEN_COLLAPSED)
      : visibleTags;
    return (
      <div>
        {tagsToDisplay.map((entry) => {
          const tagName = entry[0];
          const value = entry[1];
          return (
            <div key={tagName}>
              {tagName}:{value}
            </div>
          );
        })}
        {visibleTags.length > 3 ? (
          <a onClick={this.handleToggleCollapse}>
            {this.state.collapsed
              ? `${visibleTags.length - NUM_VISIBLE_TAGS_WHEN_COLLAPSED} more`
              : `Show less`}
          </a>
        ) : null}
      </div>
    );
  }
}
