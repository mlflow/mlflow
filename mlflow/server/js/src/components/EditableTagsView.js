import React from 'react';
import { connect } from 'react-redux';
import HtmlTableView from './HtmlTableView';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button, message } from 'antd';
import { getUUID, setTagApi } from '../Actions';
import { EditableFormTable } from './tables/EditableFormTable';

class EditableTagsView extends React.Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    tableStyles: PropTypes.object.isRequired,
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
  };

  state = { isRequestPending: false };

  tableColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      width: 200,
    },
    {
      title: 'Value',
      dataIndex: 'value',
      width: 200,
      editable: true,
    }
  ];

  requestId = getUUID();

  getData = () => {
    const { tags } = this.props;
    return Utils.getVisibleTagValues(tags).map((values) => ({
      key: values[0],
      name: values[0],
      value: values[1],
    }));
  };

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
    const { form } = this.props;
    const { getFieldDecorator } = form;
    const { isRequestPending } = this.state;

    return (
      <div>
        <EditableFormTable
          columns={this.tableColumns}
          data={this.getData()}
        />
        <h2 style={{ marginTop: 20 }}>Add Tag</h2>
        <div style={{ marginBottom: 20 }}>
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

const mapDispatchToProps = { setTagApi };

export default connect(undefined, mapDispatchToProps)(Form.create()(EditableTagsView));