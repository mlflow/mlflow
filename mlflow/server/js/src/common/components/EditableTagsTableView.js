import React from 'react';
import Utils from '../utils/Utils';
import PropTypes from 'prop-types';
import { Form, Input, Button } from 'antd/lib/index';
import { EditableFormTable } from './tables/EditableFormTable';
import _ from 'lodash';
import { Spacer } from '../../shared/building_blocks/Spacer';
import { FormattedMessage, injectIntl } from 'react-intl';

export class EditableTagsTableViewImpl extends React.Component {
  static propTypes = {
    tags: PropTypes.object.isRequired,
    form: PropTypes.object.isRequired,
    handleAddTag: PropTypes.func.isRequired,
    handleSaveEdit: PropTypes.func.isRequired,
    handleDeleteTag: PropTypes.func.isRequired,
    isRequestPending: PropTypes.bool.isRequired,
    intl: PropTypes.any,
  };

  tableColumns = [
    {
      title: this.props.intl.formatMessage({
        defaultMessage: 'Name',
        description: 'Column title for name column in editable tags table view in MLflow',
      }),
      dataIndex: 'name',
      width: 200,
    },
    {
      title: this.props.intl.formatMessage({
        defaultMessage: 'Value',
        description: 'Column title for value column in editable tags table view in MLflow',
      }),
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
    callback(
      tagNamesSet.has(value)
        ? this.props.intl.formatMessage(
            {
              defaultMessage: 'Tag "{value}" already exists.',
              description: 'Validation message for tags that already exist in tags table in MLflow',
            },
            {
              value: value,
            },
          )
        : undefined,
    );
  };

  render() {
    const { form, isRequestPending, handleSaveEdit, handleDeleteTag, handleAddTag } = this.props;
    const { getFieldDecorator } = form;

    return (
      <Spacer direction='vertical' size='small'>
        <EditableFormTable
          columns={this.tableColumns}
          data={this.getData()}
          onSaveEdit={handleSaveEdit}
          onDelete={handleDeleteTag}
        />
        <div style={styles.addTagForm.wrapper}>
          <Form layout='inline' onSubmit={handleAddTag}>
            <Form.Item>
              {getFieldDecorator('name', {
                rules: [
                  {
                    required: true,
                    message: this.props.intl.formatMessage({
                      defaultMessage: 'Name is required.',
                      description:
                        'Error message for name requirement in editable tags table view in MLflow',
                    }),
                  },
                  { validator: this.tagNameValidator },
                ],
              })(
                <Input
                  aria-label='tag name'
                  placeholder={this.props.intl.formatMessage({
                    defaultMessage: 'Name',
                    description:
                      'Default text for name placeholder in editable tags table form in MLflow',
                  })}
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
                  placeholder={this.props.intl.formatMessage({
                    defaultMessage: 'Value',
                    description:
                      'Default text for value placeholder in editable tags table form in MLflow',
                  })}
                  style={styles.addTagForm.valueInput}
                />,
              )}
            </Form.Item>
            <Form.Item>
              <Button loading={isRequestPending} htmlType='submit' data-test-id='add-tag-button'>
                <FormattedMessage
                  defaultMessage='Add'
                  description='Add button text in editable tags table view in MLflow'
                />
              </Button>
            </Form.Item>
          </Form>
        </div>
      </Spacer>
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

export const EditableTagsTableView = injectIntl(Form.create()(EditableTagsTableViewImpl));
