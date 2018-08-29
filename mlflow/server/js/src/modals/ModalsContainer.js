import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import Immutable from 'immutable';

import { getCurrentModal, getPreviousModal } from './reducer';
import { hideModal } from './actions';


export class ModalsContainer extends Component {

  static propTypes = {
    modalComponents: PropTypes.object,
    children: PropTypes.node,

    //connected
    currentModalState: PropTypes.instanceOf(Immutable.Map),
    previousModalState: PropTypes.instanceOf(Immutable.Map),

    // actions
    hideModal: PropTypes.func,

  }


  handleClose = () => {
    this.props.hideModal()
  }

  renderModals() {
    const { modalComponents, currentModalState, previousModalState } = this.props;
    const result = [
      [currentModalState, true],
      [previousModalState, false],
    ].map(([modalState, open]) => {
      const modalName = modalState.get('modalName');
      const component = modalComponents[modalName];
      if (component) {
        return React.createElement(component, {
          key: modalName,
          open,
          onClose: this.handleClose,
          modalParams: modalState.get('modalParams')
        });
      }
      return undefined;
    }).filter(Boolean);

    return result;
  }

  render() {
    const { style, children } = this.props;
    return (
      <div>
        {this.renderModals()}
        {children}
      </div>
    );
  }
}

function mapStateToProps(state) {
  return {
    currentModalState: getCurrentModal(state),
    previousModalState: getPreviousModal(state),
  }
}

export default connect(mapStateToProps, { hideModal })(ModalsContainer)
