import React from 'react';
import { connect } from 'react-redux';
import HtmlTableView from './HtmlTableView';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button, message } from 'antd';
import { setTagApi } from '../Actions';

class EditableTagsView extends React.Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    tableStyles: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
  };

  state = { requestPending: false };

  handleAddTag = (e) => {
    e.preventDefault();
    const { form, runUuid, setTagApi } = this.props;
    form.validateFields((err, values) => {
      if (!err) {
        this.setState({ requestPending: true });
        setTagApi(runUuid, values.name, values.value)
          .then(() => {
            this.setState({ requestPending: false });
            form.resetFields();
            message.success('Tag added successfully.');
          })
          .catch((e) => {
            this.setState({ requestPending: false });
            console.error(e);
            message.error('Failed to add tag.');
          });
      }
    });
  };

  render() {
    const { tags, tableStyles, form } = this.props;
    const { getFieldDecorator } = form;
    return (
      <div>
        <HtmlTableView
          columns={["Name", "Value"]}
          values={Utils.getVisibleTagValues(tags)}
          styles={tableStyles}
        />
        <h2>Add tag</h2>
        <div>
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
              <Button htmlType='submit'>Add</Button>
            </Form.Item>
          </Form>
        </div>
        <br/> {/* TODO(Zangr) remove br  */}
      </div>
    );
  }
}

const mapDispatchToProps = { setTagApi };

export default connect(undefined, mapDispatchToProps)(Form.create()(EditableTagsView));