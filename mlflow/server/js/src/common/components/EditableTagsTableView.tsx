/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import Utils from '../utils/Utils';
import { LegacyForm, Input, Button, Spacer } from '@databricks/design-system';
import { EditableFormTable } from './tables/EditableFormTable';
import { sortBy } from 'lodash';
import { FormattedMessage, injectIntl } from 'react-intl';

type Props = {
  tags: any;
  handleAddTag: (...args: any[]) => any;
  handleSaveEdit: (...args: any[]) => any;
  handleDeleteTag: (...args: any[]) => any;
  isRequestPending: boolean;
  intl: {
    formatMessage: (...args: any[]) => any;
  };
  innerRef?: any;
};

class EditableTagsTableViewImpl extends Component<Props> {
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
    sortBy(
      Utils.getVisibleTagValues(this.props.tags).map((values) => ({
        key: values[0],
        name: values[0],
        value: values[1],
      })),
      'name',
    );

  getTagNamesAsSet = () => new Set(Utils.getVisibleTagValues(this.props.tags).map((values) => values[0]));

  tagNameValidator = (rule: any, value: any, callback: any) => {
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
    const { isRequestPending, handleSaveEdit, handleDeleteTag, handleAddTag, innerRef } = this.props;

    return (
      <>
        <EditableFormTable
          columns={this.tableColumns}
          data={this.getData()}
          onSaveEdit={handleSaveEdit}
          onDelete={handleDeleteTag}
        />
        <Spacer size="sm" />
        <div>
          {/* @ts-expect-error TS(2322): Type '{ children: Element[]; ref: any; layout: "in... Remove this comment to see the full error message */}
          <LegacyForm ref={innerRef} layout="inline" onFinish={handleAddTag} css={styles.form}>
            <LegacyForm.Item
              name="name"
              rules={[
                {
                  required: true,
                  message: this.props.intl.formatMessage({
                    defaultMessage: 'Name is required.',
                    description: 'Error message for name requirement in editable tags table view in MLflow',
                  }),
                },
                {
                  validator: this.tagNameValidator,
                },
              ]}
            >
              <Input
                componentId="codegen_mlflow_app_src_common_components_editabletagstableview.tsx_107"
                aria-label="tag name"
                data-testid="tags-form-input-name"
                placeholder={this.props.intl.formatMessage({
                  defaultMessage: 'Name',
                  description: 'Default text for name placeholder in editable tags table form in MLflow',
                })}
              />
            </LegacyForm.Item>
            <LegacyForm.Item name="value" rules={[]}>
              <Input
                componentId="codegen_mlflow_app_src_common_components_editabletagstableview.tsx_117"
                aria-label="tag value"
                data-testid="tags-form-input-value"
                placeholder={this.props.intl.formatMessage({
                  defaultMessage: 'Value',
                  description: 'Default text for value placeholder in editable tags table form in MLflow',
                })}
              />
            </LegacyForm.Item>
            <LegacyForm.Item>
              <Button
                componentId="codegen_mlflow_app_src_common_components_editabletagstableview.tsx_127"
                loading={isRequestPending}
                htmlType="submit"
                data-testid="add-tag-button"
              >
                <FormattedMessage
                  defaultMessage="Add"
                  description="Add button text in editable tags table view in MLflow"
                />
              </Button>
            </LegacyForm.Item>
          </LegacyForm>
        </div>
      </>
    );
  }
}

const styles = {
  form: (theme: any) => ({
    '& > div': { marginRight: theme.spacing.sm },
  }),
};

// @ts-expect-error TS(2769): No overload matches this call.
export const EditableTagsTableView = injectIntl(EditableTagsTableViewImpl);
