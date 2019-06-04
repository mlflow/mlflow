import React from 'react';
import { Table, Input, InputNumber, Popconfirm, Form } from 'antd';
import PropTypes from 'prop-types';

import './EditableFormTable.css';

const EditableContext = React.createContext();

class EditableCell extends React.Component {
  static propTypes = {
    editing: PropTypes.bool.isRequired,
    dataIndex: PropTypes.string.isRequired,
    title: PropTypes.string.isRequired,
    record: PropTypes.object.isRequired,
    index: PropTypes.number.isRequired,
    onSaveEdit: PropTypes.func.isRequired,
    children: PropTypes.object,
  };

  static defaultProps = {
    style: {},
  };

  renderCell = ({ getFieldDecorator }) => {
    const {
      editing,
      dataIndex,
      title,
      record,
      index,
      children,
      ...restProps
    } = this.props;
    return (
      <td {...restProps} className={editing ? 'editing-cell' : ''}>
        {editing ? (
          <Form.Item style={{ margin: 0 }}>
            {getFieldDecorator(dataIndex, {
              rules: [
                {
                  required: true,
                  message: `Please Input ${title}!`,
                },
              ],
              initialValue: record[dataIndex],
            })(<Input />)}
          </Form.Item>
        ) : (
          children
        )}
      </td>
    );
  };

  render() {
    return <EditableContext.Consumer>{this.renderCell}</EditableContext.Consumer>;
  }
}

class EditableTable extends React.Component {
  static propTypes = {
    /*
      {
        title: 'name',
        dataIndex: 'name',
        width: '25%',
        editable: true,
      },
     */
    columns: PropTypes.arrayOf(Object).isRequired,
    /*
      {
        key: i.toString(),
        name: `Edrward ${i}`,
        age: 32,
        address: `London Park no. ${i}`,
      }
    */
    data: PropTypes.arrayOf(Object).isRequired,
  };

  constructor(props) {
    super(props);
    this.state = { editingKey: '' };
    this.columns = this.initColumns();
  }

  initColumns = () => [
    ...this.props.columns,
    {
      title: 'Operation',
      dataIndex: 'operation',
      width: 100,
      render: (text, record) => {
        const { editingKey } = this.state;
        const editable = this.isEditing(record);
        return editable ? (
          <span>
            <EditableContext.Consumer>
              {(form) => (
                <a onClick={() => this.save(form, record.key)} style={{ marginRight: 8 }}>
                  Save
                </a>
              )}
            </EditableContext.Consumer>
            <a onClick={() => this.cancel(record.key)}>Cancel</a>
          </span>
        ) : (
          <a disabled={editingKey !== ''} onClick={() => this.edit(record.key)}>
            Edit
          </a>
        );
      },
    },
  ];

  isEditing = (record) => record.key === this.state.editingKey;

  cancel = () => {
    this.setState({ editingKey: '' });
  };

  save(form, key) {
    form.validateFields((err, row) => {
      if (!err) {
        const dataRow = this.props.data.find((row) => row.key === key);
        if (dataRow) {
          this.props.onSaveEdit({ ...dataRow, ...row });
        }
        this.setState({ editingKey: '' });
      }
    });
  };

  edit(key) {
    this.setState({ editingKey: key });
  };

  render() {
    const components = {
      body: {
        cell: EditableCell,
      },
    };

    const columns = this.columns.map(col => {
      if (!col.editable) {
        return col;
      }
      return {
        ...col,
        onCell: record => ({
          record,
          inputType: col.dataIndex || 'text',
          dataIndex: col.dataIndex,
          title: col.title,
          editing: this.isEditing(record),
        }),
      };
    });

    const { data, style } = this.props;
    return (
      <EditableContext.Provider value={this.props.form}>
        <Table
          className='editable-table'
          components={components}
          dataSource={data}
          columns={columns}
          size='middle'
          pagination={false}
        />
      </EditableContext.Provider>
    );
  }
}

export const EditableFormTable = Form.create()(EditableTable);
