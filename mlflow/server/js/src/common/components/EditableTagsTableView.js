import React from 'react';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button } from 'antd/lib/index';
import { EditableFormTable } from './tables/EditableFormTable';
import _ from 'lodash';

export class EditableTagsTableView extends React.Component {
  static propTypes = {
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
    handleAddTag: PropTypes.func.isRequired,
    handleSaveEdit: PropTypes.func.isRequired,
    handleDeleteTag: PropTypes.func.isRequired,
    isRequestPending: PropTypes.bool.isRequired,
  };

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
    },
  ];

  getData = () =>
    _.sortBy(
      Utils.getVisibleTagValues(this.props.tags).map((values) => ({
        key: values[0],
        name: values[0],
        value: values[1],
      })),
      'name',
    );

  getTagNamesAsSet = () =>
    new Set(Utils.getVisibleTagValues(this.props.tags).map((values) => values[0]));

  tagNameValidator = (rule, value, callback) => {
    const tagNamesSet = this.getTagNamesAsSet();
    callback(tagNamesSet.has(value) ? `Tag "${value}" already exists.` : undefined);
  };

  render() {
    const { form, isRequestPending, handleSaveEdit, handleDeleteTag, handleAddTag } = this.props;
    const { getFieldDecorator } = form;

    return (
      <div>
        <EditableFormTable
          columns={this.tableColumns}
          data={this.getData()}
          onSaveEdit={handleSaveEdit}
          onDelete={handleDeleteTag}
        />
        <div style={styles.addTagForm.wrapper}>
          <h2 style={styles.addTagForm.label}>Add Tag</h2>
          <Form layout='inline' onSubmit={handleAddTag}>
            <Form.Item>
              {getFieldDecorator('name', {
                rules: [
                  { required: true, message: 'Name is required.' },
                  { validator: this.tagNameValidator },
                ],
              })(
                <Input
                  aria-label='tag name'
                  placeholder='Name'
                  style={styles.addTagForm.nameInput}
                />,
              )}
            </Form.Item>
            <Form.Item>
              {getFieldDecorator('value', {
                rules: [],
              })(
                <Input
                  aria-label='tag value'
                  placeholder='Value'
                  style={styles.addTagForm.valueInput}
                />,
              )}
            </Form.Item>
            <Form.Item>
              <Button loading={isRequestPending} htmlType='submit'>
                Add
              </Button>
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
    nameInput: { width: 186 },
    valueInput: { width: 186 },
  },
};

export default Form.create()(EditableTagsTableView);
