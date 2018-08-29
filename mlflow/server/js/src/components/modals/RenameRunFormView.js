import React, { Component } from 'react'
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { Button, Modal, Form } from 'react-bootstrap';

import { Formik, Field } from 'formik';

import { validationSchema } from './validation';


/** TODO: Can we make this a generic class that renders a form? */
class RenameRunFormView extends Component {

  static propTypes = {
    onSubmit: PropTypes.func.isRequired
  }

  getInputProps(renderProps, name, label) {
    return {
      name,
      label,
      onChange: renderProps.handleChange,
      onBlur: renderProps.handleBlur,
      value: renderProps.values[name],
      error: renderProps.errors[name],
      touched: renderProps.touched[name],
    }
  }

  handleSubmit = (
    values,
    {
      props,
      setSubmitting,
      setErrors /* setValues, setStatus, and other goodies */,
    }) => {
      return this.props.onSubmit(values).catch((err) => {
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
    const { classes } = this.props;
    return <form onSubmit={handleSubmit}>
      <h2> Rename Run </h2>
      <div> Please enter a new name for the run: </div>
      <div style={{"marginTop":"8px", "width": "80%"}}>
        <Field
            type="newRunName"
            name="newRunName"
            label="New Run Name"
            autoFocus
            style={{"width": "100%"}}
        />
        <div style={{"margin-top": "8px"}}>
        <Button bsStyle="primary" type="submit" disabled={isSubmitting}>
          Save
        </Button>
        </div>
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

export default RenameRunFormView;
