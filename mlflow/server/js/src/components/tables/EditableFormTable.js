import React from 'react';
import { Table, Input, Form, Icon } from 'antd';
import PropTypes from 'prop-types';

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
  };

  static defaultProps = {
    style: {},
  };

  render() {
    const { editing, dataIndex, title, record, children, ...restProps } = this.props;
    return (
      <EditableContext.Consumer>
        {({ getFieldDecorator }) => (
          <td {...restProps} className={editing ? 'editing-cell' : ''}>
            {editing ? (
              <Form.Item style={{ margin: 0 }}>
                {getFieldDecorator(dataIndex, {
                  rules: [
                    {
                      required: true,
                      message: `${title} is required.`,
                    },
                  ],
                  initialValue: record[dataIndex],
                })(<Input />)}
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

class EditableTable extends React.Component {
  static propTypes = {
    columns: PropTypes.arrayOf(Object).isRequired,
    data: PropTypes.arrayOf(Object).isRequired,
    onSaveEdit: PropTypes.func.isRequired,
    form: PropTypes.object.isRequired,
  };

  constructor(props) {
    super(props);
    this.state = { editingKey: '', isRequestPending: false };
    this.columns = this.initColumns();
  }

  initColumns = () => [
    ...this.props.columns.map((col) =>
      col.editable
        ? {
          ...col,
          // onCell returns props to be added to EditableCell
          onCell: (record) => ({
            record,
            dataIndex: col.dataIndex,
            title: col.title,
            editing: this.isEditing(record),
          }),
        }
        : col,
    ),
    {
      title: 'Actions',
      dataIndex: 'operation',
      width: 100,
      render: (text, record) => {
        const { editingKey, isRequestPending } = this.state;
        const editing = this.isEditing(record);
        if (editing && isRequestPending) {
          return <Icon type='loading' />;
        }
        return editing ? (
          <span>
            <EditableContext.Consumer>
              {(form) => (
                <a onClick={() => this.save(form, record.key)} style={{ marginRight: 10 }}>
                  Save
                </a>
              )}
            </EditableContext.Consumer>
            <a onClick={() => this.cancel(record.key)}>Cancel</a>
          </span>
        ) : (
          <a disabled={editingKey !== ''} onClick={() => this.edit(record.key)}>
            <Icon type='edit' />
          </a>
        );
      },
    },
  ];

  isEditing = (record) => record.key === this.state.editingKey;

  cancel = () => {
    this.setState({ editingKey: '' });
  };

  save = (form, key) => {
    form.validateFields((err, values) => {
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
        />
      </EditableContext.Provider>
    );
  }
}

export const EditableFormTable = Form.create()(EditableTable);
