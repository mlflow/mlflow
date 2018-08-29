import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Immutable from 'immutable';

//import Dialog from '@material-ui/core/Dialog';
//import DialogContent from '@material-ui/core/DialogContent';
//import DialogTitle from '@material-ui/core/DialogTitle';


import Utils from '../../utils/Utils';

import { Button, Modal } from 'react-bootstrap';

import RequestStateWrapper from '../RequestStateWrapper';



class RenameRunFailedModal extends Component {

  static propTypes = {
    modalParams: PropTypes.instanceOf(Immutable.Map),
    open: PropTypes.bool,
    onClose: PropTypes.func.isRequired,
  }

  render() {
    const { open, onClose } = this.props;
    return (<Modal show={open} onHide={onClose} bsSize="small">
      <Modal.Header> Unable to update run name. </Modal.Header>
      <Modal.Body>
        <div> Please check your network connection and try again. </div>
        <Button onClick={onClose} style={{"marginTop": "8px"}}>Ok</Button>
      </Modal.Body>
    </Modal>);
  }
}

export default RenameRunFailedModal
