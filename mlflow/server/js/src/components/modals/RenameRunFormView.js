import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Button, Modal, Form } from 'react-bootstrap';

import { Formik, Field } from 'formik';

import { validationSchema } from './validation';
import { showModal } from '../../modals/actions';

/** TODO: Can we make this a generic class that renders a form? */
class RenameRunFormView extends Component {

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
      return this.props.onSubmit(values).catch((err) => {
        this.props.showModal('RenameRunFailedModal', {});
        setErrors(err.errors)
      }).finally(() => {
        setSubmitting(false)
      })
    }

  renderForm = (renderProps) => {
    const {
      handleSubmit,
      isSubmitting,
    } = renderProps;
    return <form onSubmit={handleSubmit}>
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
      <div style={{"display": "flex", "justify-content": "flex-end"}}>
        <Button bsStyle="primary" type="submit" className="save-button" disabled={isSubmitting}>
          Save
        </Button>
        <Button bsStyle="default" type="submit" className="cancel-button" disabled={isSubmitting} onClick={this.props.onCancel}>
          Cancel
        </Button>
      </div>
    </form>;
  }

  render() {
    const { initialValues } = this.props;
    return <div>
      <Formik
        initialValues={initialValues}
        validationSchema={validationSchema}
        onSubmit={this.handleSubmit}
        render={this.renderForm}/>
    </div>
  }
}

export default connect(function() {}, { showModal })(RenameRunFormView);
