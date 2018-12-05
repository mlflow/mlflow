import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Dropdown } from 'react-bootstrap';
import RootCloseWrapper from 'react-overlays/lib/RootCloseWrapper';


export default class EmptyIfClosedMenu extends Component {
  static propTypes = {
    children: PropTypes.array.isRequired,
    open: PropTypes.bool.isRequired,
    rootCloseEvent: PropTypes.oneOf(['click', 'mousedown']),
  };

  render() {
    const {children, open, ...props} = this.props;
      // console.log("Am I open?: " + open);
      if (!open) {
        return null;
      }
      return (
        <RootCloseWrapper onRootClose={this.props.onClose}>
          <Dropdown.Menu {...props} >
            {children}
          </Dropdown.Menu>
        </RootCloseWrapper>
      );
  }

}
