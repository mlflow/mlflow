import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { css } from 'emotion';

class ExpandableList extends Component {
  state = {
    toggled: false,
  };

  static propTypes = {
    children: PropTypes.array.isRequired,
    onToggle: PropTypes.func,
    // how many lines to show in the compressed state
    showLines: PropTypes.number,
  };

  static defaultProps = {
    showLines: 1,
  };

  handleToggle = () => {
    this.setState((prevState) => ({
      toggled: !prevState.toggled,
    }));
    if (this.props.onToggle) {
      this.props.onToggle();
    }
  };

  render() {
    if (this.props.children.length <= this.props.showLines) {
      return (
        <div className={expandableListClassName}>
          {this.props.children.map((item) => (
            <div className='expandable-list-item'>{item}</div>
          ))}
        </div>
      );
    } else {
      const expandedElems = this.props.children
        .slice(this.props.showLines)
        .map((item) => <div className='expandable-list-item'>{item}</div>);
      const expandedContent = (
        <div className='expanded-list-elems'>
          {expandedElems}
          <div onClick={this.handleToggle} className='expander-text'>
            Less
          </div>
        </div>
      );
      const showMore = (
        <div onClick={this.handleToggle} className='expander-text'>
          +{this.props.children.length - this.props.showLines} more
        </div>
      );
      return (
        <div className={expandableListClassName}>
          {this.props.children.slice(0, this.props.showLines).map((item) => (
            <div className='expandable-list-item'>{item}</div>
          ))}
          {this.state.toggled ? expandedContent : showMore}
        </div>
      );
    }
  }
}

const expandableListClassName = css({
  '.expander-text': {
    textDecoration: 'underline',
    cursor: 'pointer',
  },
});

export default ExpandableList;
