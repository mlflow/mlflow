/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';

type Props = {
  onToggle?: (...args: any[]) => any;
  showLines?: number;
};

type State = any;

class ExpandableList extends Component<Props, State> {
  state = {
    toggled: false,
  };

  static defaultProps = {
    showLines: 1,
  };

  handleToggle = () => {
    this.setState((prevState: any) => ({
      toggled: !prevState.toggled,
    }));
    if (this.props.onToggle) {
      this.props.onToggle();
    }
  };

  render() {
    if ((this.props.children as any).length <= (this.props.showLines ?? 1)) {
      return (
        <div css={expandableListClassName}>
          {(this.props.children as any).map((item: any, index: any) => (
            <div className="expandable-list-item" key={index}>
              {item}
            </div>
          ))}
        </div>
      );
    } else {
      const expandedElems = (this.props.children as any).slice(this.props.showLines).map((item: any, index: any) => (
        <div className="expandable-list-item" key={index}>
          {item}
        </div>
      ));
      const expandedContent = (
        <div className="expanded-list-elems">
          {expandedElems}
          <div onClick={this.handleToggle} className="expander-text">
            Less
          </div>
        </div>
      );
      const showMore = (
        <div onClick={this.handleToggle} className="expander-text">
          +{(this.props.children as any).length - (this.props.showLines ?? 1)} more
        </div>
      );
      return (
        <div css={expandableListClassName}>
          {(this.props.children as any).slice(0, this.props.showLines).map((item: any, index: any) => (
            <div className="expandable-list-item" key={index}>
              {item}
            </div>
          ))}
          {this.state.toggled ? expandedContent : showMore}
        </div>
      );
    }
  }
}

const expandableListClassName = {
  '.expander-text': {
    textDecoration: 'underline',
    cursor: 'pointer',
  },
};

export default ExpandableList;
