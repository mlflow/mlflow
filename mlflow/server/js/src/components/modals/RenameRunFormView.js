import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Button, Modal, Form } from 'react-bootstrap';

import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
import { showModal } from '../../modals/actions';

import onClickOutside from "react-onclickoutside";


class RenameRunFormView extends Component {

  constructor(props) {
    super(props);
    this.handleClickOutside = this.handleClickOutside.bind(this);
  }

  handleClickOutside(event) {
    console.log(event);
    console.log(this.state.isSubmittingState);
    if (this.state.isSubmittingState) {
      event.preventDefault();
      event.stopPropagation();
    }
  };

  state = {
    isSubmittingState: false,
  };

  static propTypes = {
    onSubmit: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
  }

  handleSubmit = (
    values,
    {
      props,
      setSubmitting,
      setErrors /* setValues, setStatus, and other goodies */,
    }) => {
      this.setState({isSubmittingState: true});
      return this.props.onSubmit(values).catch((err) => {
        // TODO: Handle errors here: on network failures, show failed form. If run was deleted,
        // redirect to an error page.
        setErrors(err.errors)
      }).finally(function() {
        this.setState({isSubmittingState: false});
        setSubmitting(false);
        this.props.onCancel();
      }.bind(this))
    }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
    } = renderProps;
    return <form onSubmit={handleSubmit} style={{"width": "480px"}}>
      <h2 style={{"marginTop": "0px"}}> Rename Run </h2>
      <div style={{"marginTop": "16px", "marginBottom": "16px"}}> New run name: </div>
      <div style={{"width": "100%", "marginBottom": "16px"}}>
        <Field
            type="newRunName"
            name="newRunName"
            label="New Run Name"
            autoFocus
            style={{"width": "100%"}}
        />
      </div>
      <div style={{"display": "flex", "justifyContent": "flex-end"}}>
        <Button bsStyle="primary" type="submit" className="save-button" disabled={isSubmitting}>
          Save
        </Button>
        <Button bsStyle="default" className="cancel-button" disabled={isSubmitting} onClick={this.props.onCancel}>
          Cancel
        </Button>
      </div>
    </form>;
  }

  render() {
    return <div>
      <Formik
        initialValues={{newRunName: ""}}
        validationSchema={validationSchema}
        onSubmit={this.handleSubmit}
        render={this.renderForm}/>
    </div>
  }
}

export default connect(function() {return {}}, { showModal })(RenameRunFormView);
