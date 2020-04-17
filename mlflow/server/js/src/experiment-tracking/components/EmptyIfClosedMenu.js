import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Dropdown } from 'react-bootstrap';
import RootCloseWrapper from 'react-overlays/lib/RootCloseWrapper';


export default class EmptyIfClosedMenu extends Component {
  static propTypes = {
    children: PropTypes.array.isRequired,
    open: PropTypes.bool,
    onClose: PropTypes.func,
  };

  render() {
    const {children, open, onClose, ...props} = this.props;
    if (!open) {
      return null;
    }
    return (
      <RootCloseWrapper onRootClose={onClose}>
        <Dropdown.Menu {...props} >
          {children}
        </Dropdown.Menu>
      </RootCloseWrapper>
    );
  }
}
