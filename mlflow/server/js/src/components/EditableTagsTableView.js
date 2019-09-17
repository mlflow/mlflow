import React from 'react';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button, message } from 'antd';
import { getUUID, setTagApi } from '../Actions';
import { EditableFormTable } from './tables/EditableFormTable';
import _ from 'lodash';

export class EditableTagsTableView extends React.Component {
  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
    setTagApi: PropTypes.func.isRequired,
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

  getData = () => _.sortBy(Utils.getVisibleTagValues(this.props.tags).map((values) => ({
    key: values[0],
    name: values[0],
    value: values[1],
  })), 'name');

  getTagNamesAsSet = () => new Set(
    Utils.getVisibleTagValues(this.props.tags).map((values) => values[0])
  );

  handleAddTag = (e) => {
    e.preventDefault();
    const { form, runUuid, setTagApi: setTag } = this.props;
    form.validateFields((err, values) => {
      if (!err) {
        this.setState({ isRequestPending: true });
        setTag(runUuid, values.name, values.value, this.requestId)
          .then(() => {
            this.setState({ isRequestPending: false });
            form.resetFields();
          })
          .catch((ex) => {
            this.setState({ isRequestPending: false });
            console.error(ex);
            message.error('Failed to add tag. Error: ' + ex.getUserVisibleError());
          });
      }
    });
  };

  handleSaveEdit = ({ name, value }) => {
    const { runUuid, setTagApi: setTag } = this.props;
    return setTag(runUuid, name, value, this.requestId)
      .catch((ex) => {
        console.error(ex);
        message.error('Failed to set tag. Error: ' + ex.getUserVisibleError());
      });
  };

  tagNameValidator = (rule, value, callback) => {
    const tagNamesSet = this.getTagNamesAsSet();
    callback(tagNamesSet.has(value) ? `Tag "${value}" already exists.` : undefined);
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
          onSaveEdit={this.handleSaveEdit}
        />
        <div style={styles.addTagForm.wrapper}>
          <h2 style={styles.addTagForm.label}>Add Tag</h2>
          <Form layout='inline' onSubmit={this.handleAddTag} style={styles.addTagForm.form}>
            <Form.Item>
              {getFieldDecorator('name', {
                rules: [
                  { required: true, message: 'Name is required.'},
                  { validator: this.tagNameValidator },
                ],
              })(
                <Input placeholder='Name' style={styles.addTagForm.nameInput}/>
              )}
            </Form.Item>
            <Form.Item>
              {getFieldDecorator('value', {
                rules: []
              })(
                <Input placeholder='Value' style={styles.addTagForm.valueInput}/>
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
    wrapper: { marginLeft: 7 },
    label: {
      marginTop: 20,
    },
    form: { marginBottom: 20 },
    nameInput: { width: 186 },
    valueInput: { width: 186 },
  }
};

const mapDispatchToProps = { setTagApi };

export default connect(undefined, mapDispatchToProps)(Form.create()(EditableTagsTableView));
