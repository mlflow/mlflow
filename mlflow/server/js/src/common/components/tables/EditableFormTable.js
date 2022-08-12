import React from 'react';
import {
  Input,
  Button,
  Form,
  Modal,
  Table,
  PencilIcon,
  Spinner,
  TrashIcon,
} from '@databricks/design-system';
import PropTypes from 'prop-types';
import { FormattedMessage } from 'react-intl';

const EditableContext = React.createContext();

class EditableCell extends React.Component {
  static propTypes = {
    editing: PropTypes.bool,
    dataIndex: PropTypes.string,
    title: PropTypes.string,
    record: PropTypes.object,
    index: PropTypes.number,
    children: PropTypes.oneOfType([PropTypes.object, PropTypes.array, PropTypes.node]),
    save: PropTypes.func,
    cancel: PropTypes.func,
    recordKey: PropTypes.string,
  };

  handleKeyPress = (event) => {
    const { save, recordKey, cancel } = this.props;
    if (event.key === 'Enter') {
      save(recordKey);
    } else if (event.key === 'Escape') {
      cancel();
    }
  };

  render() {
    const { editing, dataIndex, record, children } = this.props;
    return (
      <EditableContext.Consumer>
        {({ formRef }) => (
          <div className={editing ? 'editing-cell' : ''}>
            {editing ? (
              <Form ref={formRef}>
                <Form.Item style={{ margin: 0 }} name={dataIndex} initialValue={record[dataIndex]}>
                  <Input onKeyDown={this.handleKeyPress} />
                </Form.Item>
              </Form>
            ) : (
              children
            )}
          </div>
        )}
      </EditableContext.Consumer>
    );
  }
}

export class EditableTable extends React.Component {
  static propTypes = {
    columns: PropTypes.arrayOf(PropTypes.object).isRequired,
    data: PropTypes.arrayOf(PropTypes.object).isRequired,
    onSaveEdit: PropTypes.func.isRequired,
    onDelete: PropTypes.func.isRequired,
    intl: PropTypes.any,
  };

  constructor(props) {
    super(props);
    this.state = { editingKey: '', isRequestPending: false, deletingKey: '' };
    this.columns = this.initColumns();
    this.form = React.createRef();
  }

  initColumns = () => [
    ...this.props.columns.map((col) =>
      col.editable
        ? {
            ...col,
            render: (text, record) => (
              <EditableCell
                record={record}
                dataIndex={col.dataIndex}
                title={col.title}
                editing={this.isEditing(record)}
                save={this.save}
                cancel={this.cancel}
                recordKey={record.key}
                children={text}
              />
            ),
          }
        : col,
    ),
    {
      title: (
        <FormattedMessage
          defaultMessage='Actions'
          description='Column title for actions column in editable form table in MLflow'
        />
      ),
      dataIndex: 'operation',
      render: (text, record) => {
        const { editingKey, isRequestPending } = this.state;
        const editing = this.isEditing(record);
        if (editing && isRequestPending) {
          return <Spinner size='small' />;
        }
        return editing ? (
          <span>
            <Button type='link' onClick={() => this.save(record.key)} style={{ marginRight: 10 }}>
              <FormattedMessage
                defaultMessage='Save'
                description='Text for saving changes on rows in editable form table in MLflow'
              />
            </Button>
            <Button type='link' onClick={() => this.cancel(record.key)}>
              <FormattedMessage
                defaultMessage='Cancel'
                description='Text for canceling changes on rows in editable form table in MLflow'
              />
            </Button>
          </span>
        ) : (
          <span>
            <Button
              icon={<PencilIcon />}
              disabled={editingKey !== ''}
              onClick={() => this.edit(record.key)}
            />
            <Button
              icon={<TrashIcon />}
              disabled={editingKey !== ''}
              onClick={() => this.setState({ deletingKey: record.key })}
            />
          </span>
        );
      },
    },
  ];

  isEditing = (record) => record.key === this.state.editingKey;

  cancel = () => {
    this.setState({ editingKey: '' });
  };

  save = (key) => {
    this.form.current.validateFields().then((values) => {
      const record = this.props.data.find((r) => r.key === key);
      if (record) {
        this.setState({ isRequestPending: true });
        this.props.onSaveEdit({ ...record, ...values }).then(() => {
          this.setState({ editingKey: '', isRequestPending: false });
        });
      }
    });
  };

  delete = async (key) => {
    try {
      const record = this.props.data.find((r) => r.key === key);
      if (record) {
        this.setState({ isRequestPending: true });
        await this.props.onDelete({ ...record });
      }
    } finally {
      this.setState({ deletingKey: '', isRequestPending: false });
    }
  };

  edit = (key) => {
    this.setState({ editingKey: key });
  };

  render() {
    const { data } = this.props;
    return (
      <EditableContext.Provider value={{ formRef: this.form }}>
        <Table
          className='editable-table'
          data-testid='editable-table'
          dataSource={data}
          columns={this.columns}
          size='middle'
          tableLayout='fixed'
          pagination={false}
          locale={{
            emptyText: (
              <FormattedMessage
                defaultMessage='No tags found.'
                description='Text for no tags found in editable form table in MLflow'
              />
            ),
          }}
          scroll={{ y: 280 }}
        />
        <Modal
          data-testid='editable-form-table-remove-modal'
          title={
            <FormattedMessage
              defaultMessage='Are you sure you want to delete this tag？'
              description='Title text for confirmation pop-up to delete a tag from table
                     in MLflow'
            />
          }
          visible={this.state.deletingKey}
          okText={
            <FormattedMessage
              defaultMessage='Confirm'
              description='OK button text for confirmation pop-up to delete a tag from table
                     in MLflow'
            />
          }
          cancelText={
            <FormattedMessage
              defaultMessage='Cancel'
              description='Cancel button text for confirmation pop-up to delete a tag from
                     table in MLflow'
            />
          }
          confirmLoading={this.state.isRequestPending}
          onOk={() => this.delete(this.state.deletingKey)}
          onCancel={() => this.setState({ deletingKey: '' })}
        />
      </EditableContext.Provider>
    );
  }
}

export const EditableFormTable = EditableTable;
