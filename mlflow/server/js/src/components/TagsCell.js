import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../utils/Utils';

export class TagsCell extends React.Component {
  static propTypes = {
    tags: PropTypes.object.isRequired,
  };

  state = {
    collapsed: true
  };

  handleToggleCollapse = () => {
    this.setState((prevState) => ({ collapsed: !prevState.collapsed }));
  };

  render() {
    const visibleTags = Utils.getVisibleTagValues(this.props.tags);
    const tagsToDisplay = this.state.collapsed ? visibleTags.slice(0, 3) : visibleTags;
    return (
      <div>
        {tagsToDisplay.map((entry) => {
          const tagName = entry[0];
          const value = entry[1];
          return <div key={tagName}>{tagName}:{value}</div>
        })}
        {visibleTags.length > 3 ? (
          <a onClick={this.handleToggleCollapse}>
            {this.state.collapsed ? `${visibleTags.length - 3} more` : `Show less`}
          </a>
        ): null}
      </div>
    );
  }
}
