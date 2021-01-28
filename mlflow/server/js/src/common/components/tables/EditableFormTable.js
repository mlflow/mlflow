import React from 'react';
import { Table, Input, Form, Icon, Popconfirm, Button } from 'antd';
import PropTypes from 'prop-types';
import { IconButton } from '../../components/IconButton';
import _ from 'lodash';

import './EditableFormTable.css';

const EditableContext = React.createContext();

class EditableCell extends React.Component {
  static propTypes = {
    editing: PropTypes.bool,
    dataIndex: PropTypes.string,
    title: PropTypes.string,
    record: PropTypes.object,
    index: PropTypes.number,
    children: PropTypes.oneOfType([PropTypes.object, PropTypes.array]),
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
        {({ getFieldDecorator }) => (
          <td className={editing ? 'editing-cell' : ''}>
            {editing ? (
              <Form.Item style={{ margin: 0 }}>
                {getFieldDecorator(dataIndex, {
                  rules: [],
                  initialValue: record[dataIndex],
                })(<Input onKeyDown={this.handleKeyPress} />)}
              </Form.Item>
            ) : (
              children
            )}
          </td>
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
    form: PropTypes.object.isRequired,
  };

  constructor(props) {
    super(props);
    this.state = { editingKey: '', isRequestPending: false };
    this.columns = this.initColumns();
  }

  // set table width as sum of columns rather than hard coding a width
  // see ML-11973
  getTotalTableWidth = () => _.sumBy(this.columns, 'width');

  initColumns = () => [
    ...this.props.columns.map((col) =>
      col.editable
        ? {
            ...col,
            // `onCell` returns props to be added to EditableCell
            onCell: (record) => ({
              record,
              dataIndex: col.dataIndex,
              title: col.title,
              editing: this.isEditing(record),
              save: this.save,
              cancel: this.cancel,
              recordKey: record.key,
            }),
          }
        : col,
    ),
    {
      title: 'Actions',
      dataIndex: 'operation',
      width: 200,
      render: (text, record) => {
        const { editingKey, isRequestPending } = this.state;
        const editing = this.isEditing(record);
        if (editing && isRequestPending) {
          return <Icon type='loading' />;
        }
        return editing ? (
          <span>
            <Button type='link' onClick={() => this.save(record.key)} style={{ marginRight: 10 }}>
              Save
            </Button>
            <Button type='link' onClick={() => this.cancel(record.key)}>
              Cancel
            </Button>
          </span>
        ) : (
          <span>
            <IconButton
              icon={<Icon type='edit' />}
              disabled={editingKey !== ''}
              onClick={() => this.edit(record.key)}
              style={{ marginRight: 10 }}
            />
            <Popconfirm
              title='Are you sure you want to delete this tag？'
              okText='Confirm'
              cancelText='Cancel'
              onConfirm={() => this.delete(record.key)}
            >
              <IconButton icon={<i className='far fa-trash-alt' />} disabled={editingKey !== ''} />
            </Popconfirm>
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
    this.props.form.validateFields((err, values) => {
      if (!err) {
        const record = this.props.data.find((r) => r.key === key);
        if (record) {
          this.setState({ isRequestPending: true });
          this.props.onSaveEdit({ ...record, ...values }).then(() => {
            this.setState({ editingKey: '', isRequestPending: false });
          });
        }
      }
    });
  };

  delete = (key) => {
    const record = this.props.data.find((r) => r.key === key);
    if (record) {
      this.setState({ isRequestPending: true });
      this.props.onDelete({ ...record }).then(() => {
        this.setState({ editingKey: '', isRequestPending: false });
      });
    }
  };

  edit = (key) => {
    this.setState({ editingKey: key });
  };

  render() {
    const components = {
      body: {
        cell: EditableCell,
      },
    };
    const { data, form } = this.props;
    return (
      <EditableContext.Provider value={form}>
        <Table
          className='editable-table'
          components={components}
          dataSource={data}
          columns={this.columns}
          size='middle'
          pagination={false}
          locale={{ emptyText: 'No tags found.' }}
          scroll={{ y: 280 }}
          style={{ width: this.getTotalTableWidth() }}
        />
      </EditableContext.Provider>
    );
  }
}

export const EditableFormTable = Form.create()(EditableTable);
