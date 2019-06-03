import React from 'react';
import { connect } from 'react-redux';
import HtmlTableView from './HtmlTableView';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button, message } from 'antd';
import { getUUID, setTagApi } from '../Actions';

class EditableTagsView extends React.Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    tableStyles: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
  };

  state = { isRequestPending: false };

  requestId = getUUID();

  handleAddTag = (e) => {
    e.preventDefault();
    const { form, runUuid, setTagApi } = this.props;
    form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isRequestPending: true });
        setTagApi(runUuid, values.name, values.value, this.requestId)
          .then(() => {
            this.setState({ isRequestPending: false });
            form.resetFields();
            message.success('Tag added successfully.');
          })
          .catch((e) => {
            this.setState({ isRequestPending: false });
            console.error(e);
            message.error('Failed to add tag.');
          });
      }
    });
  };

  render() {
    const { tags, tableStyles, form } = this.props;
    const { getFieldDecorator } = form;
    const { isRequestPending } = this.state;
    return (
      <div>
        <HtmlTableView
          columns={["Name", "Value"]}
          values={Utils.getVisibleTagValues(tags)}
          styles={tableStyles}
        />
        <h2>Add Tag</h2>
        <div className='add-tag-form' style={styles.addTagForm}>
          <Form layout='inline' onSubmit={this.handleAddTag}>
            <Form.Item>
              {getFieldDecorator('name', {
                rules: [{ required: true, message: 'Name is required.'}]
              })(
                <Input placeholder='Name'/>
              )}
            </Form.Item>
            <Form.Item>
              {getFieldDecorator('value', {
                rules: [{ required: true, message: 'Value is required.'}]
              })(
                <Input placeholder='Value'/>
              )}
            </Form.Item>
            <Form.Item>
              <Button loading={isRequestPending} htmlType='submit'>Add</Button>
            </Form.Item>
          </Form>
        </div>
      </div>
    );
  }
}

const styles = {
  addTagForm: {
    marginBottom: 20,
  }
};

const mapDispatchToProps = { setTagApi };

export default connect(undefined, mapDispatchToProps)(Form.create()(EditableTagsView));