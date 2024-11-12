/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import {
  Input,
  Button,
  LegacyForm,
  Modal,
  LegacyTable,
  PencilIcon,
  Spinner,
  TrashIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

// @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
const EditableContext = React.createContext();

type EditableCellProps = {
  editing?: boolean;
  dataIndex?: string;
  title?: string;
  record?: any;
  index?: number;
  save?: (...args: any[]) => any;
  cancel?: (...args: any[]) => any;
  recordKey?: string;
};

class EditableCell extends React.Component<EditableCellProps> {
  handleKeyPress = (event: any) => {
    const { save, recordKey, cancel } = this.props;
    if (event.key === 'Enter') {
      // @ts-expect-error TS(2722): Cannot invoke an object which is possibly 'undefin... Remove this comment to see the full error message
      save(recordKey);
    } else if (event.key === 'Escape') {
      // @ts-expect-error TS(2722): Cannot invoke an object which is possibly 'undefin... Remove this comment to see the full error message
      cancel();
    }
  };

  render() {
    const { editing, dataIndex, record, children } = this.props;
    return (
      <EditableContext.Consumer>
        {/* @ts-expect-error TS(2322): Type '({ formRef }: { formRef: any; }) => Element'... Remove this comment to see the full error message */}
        {({ formRef }) => (
          <div className={editing ? 'editing-cell' : ''}>
            {editing ? (
              // @ts-expect-error TS(2322): Type '{ children: Element; ref: any; }' is not ass... Remove this comment to see the full error message
              <LegacyForm ref={formRef}>
                {/* @ts-expect-error TS(2322): Type '{ children: Element; style: { margin: number... Remove this comment to see the full error message */}
                <LegacyForm.Item style={{ margin: 0 }} name={dataIndex} initialValue={record[dataIndex]}>
                  <Input
                    componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_50"
                    onKeyDown={this.handleKeyPress}
                    data-testid="editable-table-edited-input"
                  />
                </LegacyForm.Item>
              </LegacyForm>
            ) : (
              children
            )}
          </div>
        )}
      </EditableContext.Consumer>
    );
  }
}

type EditableTableProps = {
  columns: any[];
  data: any[];
  onSaveEdit: (...args: any[]) => any;
  onDelete: (...args: any[]) => any;
  intl?: any;
};

type EditableTableState = any;

export class EditableTable extends React.Component<EditableTableProps, EditableTableState> {
  columns: any;
  form: any;

  constructor(props: EditableTableProps) {
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
            render: (text: any, record: any) => (
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
          defaultMessage="Actions"
          description="Column title for actions column in editable form table in MLflow"
        />
      ),
      dataIndex: 'operation',
      render: (text: any, record: any) => {
        const { editingKey, isRequestPending } = this.state;
        const editing = this.isEditing(record);
        if (editing && isRequestPending) {
          return <Spinner size="small" />;
        }
        return editing ? (
          <span>
            <Button
              componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_120"
              type="link"
              onClick={() => this.save(record.key)}
              style={{ marginRight: 10 }}
              data-testid="editable-table-button-save"
            >
              <FormattedMessage
                defaultMessage="Save"
                description="Text for saving changes on rows in editable form table in MLflow"
              />
            </Button>
            <Button
              componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_131"
              type="link"
              // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
              onClick={() => this.cancel(record.key)}
              data-testid="editable-table-button-cancel"
            >
              <FormattedMessage
                defaultMessage="Cancel"
                description="Text for canceling changes on rows in editable form table in MLflow"
              />
            </Button>
          </span>
        ) : (
          <span>
            <Button
              componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_145"
              icon={<PencilIcon />}
              disabled={editingKey !== ''}
              onClick={() => this.edit(record.key)}
              data-testid="editable-table-button-edit"
            />
            <Button
              componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_151"
              icon={<TrashIcon />}
              disabled={editingKey !== ''}
              onClick={() => this.setState({ deletingKey: record.key })}
              data-testid="editable-table-button-delete"
            />
          </span>
        );
      },
    },
  ];

  // @ts-expect-error TS(4111): Property 'editingKey' comes from an index signatur... Remove this comment to see the full error message
  isEditing = (record: any) => record.key === this.state.editingKey;

  cancel = () => {
    this.setState({ editingKey: '' });
  };

  save = (key: any) => {
    this.form.current.validateFields().then((values: any) => {
      const record = this.props.data.find((r) => r.key === key);
      if (record) {
        this.setState({ isRequestPending: true });
        this.props.onSaveEdit({ ...record, ...values }).then(() => {
          this.setState({ editingKey: '', isRequestPending: false });
        });
      }
    });
  };

  delete = async (key: any) => {
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

  edit = (key: any) => {
    this.setState({ editingKey: key });
  };

  render() {
    const { data } = this.props;
    return (
      <EditableContext.Provider value={{ formRef: this.form }}>
        <LegacyTable
          className="editable-table"
          data-testid="editable-table"
          dataSource={data}
          columns={this.columns}
          size="middle"
          tableLayout="fixed"
          pagination={false}
          locale={{
            emptyText: (
              <FormattedMessage
                defaultMessage="No tags found."
                description="Text for no tags found in editable form table in MLflow"
              />
            ),
          }}
          scroll={{ y: 280 }}
        />
        <Modal
          componentId="codegen_mlflow_app_src_common_components_tables_editableformtable.tsx_228"
          data-testid="editable-form-table-remove-modal"
          title={
            <FormattedMessage
              defaultMessage="Are you sure you want to delete this tag？"
              description="Title text for confirmation pop-up to delete a tag from table
                     in MLflow"
            />
          }
          // @ts-expect-error TS(4111): Property 'deletingKey' comes from an index signatu... Remove this comment to see the full error message
          visible={this.state.deletingKey}
          okText={
            <FormattedMessage
              defaultMessage="Confirm"
              description="OK button text for confirmation pop-up to delete a tag from table
                     in MLflow"
            />
          }
          cancelText={
            <FormattedMessage
              defaultMessage="Cancel"
              description="Cancel button text for confirmation pop-up to delete a tag from
                     table in MLflow"
            />
          }
          // @ts-expect-error TS(4111): Property 'isRequestPending' comes from an index si... Remove this comment to see the full error message
          confirmLoading={this.state.isRequestPending}
          // @ts-expect-error TS(4111): Property 'deletingKey' comes from an index signatu... Remove this comment to see the full error message
          onOk={() => this.delete(this.state.deletingKey)}
          onCancel={() => this.setState({ deletingKey: '' })}
        />
      </EditableContext.Provider>
    );
  }
}

export const EditableFormTable = EditableTable;
