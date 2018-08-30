import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Button, Modal, Form } from 'react-bootstrap';
import { withRouter } from 'react-router-dom';
import Routes from "../../Routes";
import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
class RenameRunFormView extends Component {

  constructor(props) {
    super(props);
  }

  static propTypes = {
    onSubmit: PropTypes.func.isRequired,
    onClose: PropTypes.func.isRequired,
    runName: PropTypes.string.isRequired,
    experimentId: PropTypes.number.isRequired
  }

  handleSubmit = (
    values,
    {
      props,
      setSubmitting,
      setErrors /* setValues, setStatus, and other goodies */,
    }) => {
      return this.props.onSubmit(values).catch(function(err) {
        // TODO: remove alert, redirect to an error page on failed requests once one exists
        alert("Unable to rename run, got error '" + err + "'. Redirecting to parent experiment " +
          "page.");
        this.props.history.push(Routes.getExperimentPageRoute(this.props.experimentId));
      }.bind(this)).finally(function() {
        setSubmitting(false);
        this.props.onClose();
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
        <Button bsStyle="default" className="cancel-button" disabled={isSubmitting}
          onClick={this.props.onClose}>
          Cancel
        </Button>
      </div>
    </form>;
  }

  render() {
    return <div>
      <Formik
        initialValues={{newRunName: this.props.runName}}
        validationSchema={validationSchema}
        onSubmit={this.handleSubmit}
        render={this.renderForm}/>
    </div>
  }
}

export default withRouter(RenameRunFormView);
