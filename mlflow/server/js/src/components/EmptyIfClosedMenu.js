import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { Dropdown } from 'react-bootstrap';

export default class EmptyIfClosedMenu extends Component {
  static propTypes = {
    children: PropTypes.array.isRequired,
    open: PropTypes.bool.isRequired,
  };

  render() {
    const {children, open, ...props} = this.props;
      console.log("Am I open?: " + open);
      return (
        <Dropdown.Menu {...props} >
          {open ? children : null}
        </Dropdown.Menu>
      );
  }

}
