/* eslint func-names: 0 */

import $ from 'jquery';
import React, { Component } from "react";

/**
 * Dropdown Menu Component
 *
 * Required Properties:
 *   - heading: String that will be used as the menu heading
 *   - outsideClickHandler: The handler that will be triggered when a user clicks outside of the
 *   dropdown
 *   - getItems: function that returns an array of React DOM elements
 *   It will be called to populate the dropdown menu
 *
 * Optional Properties:
 *   - classes: array of strings of class names to attach to the dropdown menu div
 *   - handleClickInMenu: call the outsideClickHandler even if the click is inside the menu div
 *   - ignoreClickClasses: don't call the outsideClickHandler if clicked on an element with one
 *     of these classes.
 *
 * TODO(?) Refactor so that all the state of the menu (up/down) is an internal state of this class
 */
export default class DropdownMenuView extends Component {

  propTypes: {
    heading: React.PropTypes.string,
    outsideClickHandler: React.PropTypes.func.isRequired,
    getItems: React.PropTypes.func.isRequired,
    handleClickInMenu: React.PropTypes.bool,
    ignoreClickClasses: React.PropTypes.array,
    classes: React.PropTypes.array,
  }

  getDefaultProps() {
    return {
      classes: [],
    };
  }

  // Set up a handler so that we hide the dropdown on outside clicks
  // We save a reference to it so that we can unbind it from the document later
  componentDidMount() {
    const self = this;
    this.documentClickHandler = function(e) {
      const ignoreClasses = ['dropdown-menu']
        .concat(self.props.ignoreClickClasses ? self.props.ignoreClickClasses : []);
//      const ignore = _.some(ignoreClasses, function(cls) {
//        return $(e.target).closest('.' + cls).length !== 0;
//      });
//      if (!ignore) {
        self.props.outsideClickHandler();
//      }
      return true;
    };

    $(document).click(this.documentClickHandler);
  }

  // Unbind the outside click handler when we remove the dropdown from the DOM
  componentWillUnmount() {
    $(document).off('click', this.documentClickHandler);
  }

  render() {
    // Wrap each passed in item in a <li>
    let key = 0;
    const items = this.props.getItems().map(function(item) {
      key++;
      return (<li key={key}>{item}</li>);
    });

    const classes = 'dropdown-custom-menu dropdown-menu ' + this.props.classes.join(' ');

    const heading = this.props.heading ? (
      <li><span className='heading'>{this.props.heading}</span></li>
    ) : null;

    const clickHandler = this.props.handleClickInMenu ? this.props.outsideClickHandler : null;

    return (
      <div className={classes} onClick={clickHandler}>
        {heading}
        {items}
      </div>
    );
  }
};
