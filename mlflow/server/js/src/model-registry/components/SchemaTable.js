import React from 'react';
import { Table } from 'antd';
import PropTypes from 'prop-types';
import { css } from 'emotion';
import { LogModelWithSignatureUrl } from '../../common/constants';
import { gray800 } from '../../common/styles/color';
import { spacingMedium } from '../../common/styles/spacing';
import { MODEL_SCHEMA_TENSOR_TYPE } from '../constants';
import { FormattedMessage, injectIntl } from 'react-intl';

const { Column } = Table;

export class SchemaTableImpl extends React.PureComponent {
  static propTypes = {
    schema: PropTypes.object,
    defaultExpandAllRows: PropTypes.bool,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  renderSchemaTable = (schemaData, schemaType) => {
    const columns = [
      {
        title: 'Name',
        dataIndex: 'name',
        key: 'name',
        width: '50%',
      },
      {
        title: 'Type',
        dataIndex: 'type',
        key: 'type',
        width: '50%',
      },
    ];

    return (
      <Table
        className='inner-table'
        size='middle'
        showHeader={false}
        pagination={false}
        locale={{ emptyText: `No schema ${schemaType}.` }}
        dataSource={this.getSchemaRowData(schemaData)}
        columns={columns}
        scroll={{ y: 240 }}
      />
    );
  };

  getSchemaTypeRepr = (schemaTypeSpec) => {
    if (schemaTypeSpec.type === MODEL_SCHEMA_TENSOR_TYPE) {
      return (
        `Tensor (dtype: ${schemaTypeSpec['tensor-spec'].dtype},` +
        ` shape: [${schemaTypeSpec['tensor-spec'].shape}])`
      );
    } else {
      return schemaTypeSpec.type;
    }
  };

  getSchemaRowData = (schemaData) => {
    const rowData = [];
    schemaData.forEach((row, index) => {
      rowData[index] = {
        key: index,
        name: row.name ? row.name : '-',
        type: row.type ? this.getSchemaTypeRepr(row) : '-',
      };
    });
    return rowData;
  };

  renderSectionHeader = (text) => {
    return <strong className='primary-text'>{text}</strong>;
  };

  render() {
    const { schema } = this.props;
    const plusIcon = <i className='far fa-plus-square' />;
    const minusIcon = <i className='far fa-minus-square' />;
    const hasSchema = schema.inputs.length || schema.outputs.length;
    const sectionHeaders = hasSchema
      ? [
          {
            key: '1',
            name: this.props.intl.formatMessage(
              {
                defaultMessage: 'Inputs ({numInputs})',
                description: 'Input section header for schema table in model version page',
              },
              {
                numInputs: schema.inputs.length,
              },
            ),
            type: '',
            table: this.renderSchemaTable(schema.inputs, 'inputs'),
          },
          {
            key: '2',
            name: this.props.intl.formatMessage(
              {
                defaultMessage: 'Outputs ({numOutputs})',
                description: 'Input section header for schema table in model version page',
              },
              {
                numOutputs: schema.outputs.length,
              },
            ),
            type: '',
            table: this.renderSchemaTable(schema.outputs, 'outputs'),
          },
        ]
      : [];

    return (
      <div className={`${schemaTableClassName}`}>
        <Table
          key='schema-table'
          className='outer-table'
          rowClassName='section-header-row'
          size='middle'
          pagination={false}
          defaultExpandAllRows={this.props.defaultExpandAllRows}
          expandRowByClick
          expandedRowRender={(record) => record.table}
          expandIcon={({ expanded }) => (expanded ? minusIcon : plusIcon)}
          locale={{
            emptyText: (
              <div>
                {/* eslint-disable-next-line max-len */}
                <FormattedMessage
                  defaultMessage='No schema. See <link>MLflow docs</link> for how to include
                     input and output schema with your model.'
                  description='Text for schema table when no schema exists in the model version
                     page'
                  values={{
                    link: (chunks) => <a href={LogModelWithSignatureUrl}>{chunks}</a>,
                  }}
                />
              </div>
            ),
          }}
          dataSource={sectionHeaders}
          scroll={{ x: 240 }}
        >
          <Column
            key={1}
            title={this.props.intl.formatMessage({
              defaultMessage: 'Name',
              description: 'Text for name column in schema table in model version page',
            })}
            width='50%'
            dataIndex='name'
            render={this.renderSectionHeader}
          />
          <Column
            key={2}
            title={this.props.intl.formatMessage({
              defaultMessage: 'Type',
              description: 'Text for type column in schema table in model version page',
            })}
            width='50%'
            dataIndex='type'
            render={this.renderSectionHeader}
          />
        </Table>
      </div>
    );
  }
}

export const SchemaTable = injectIntl(SchemaTableImpl);

const antTable = '.ant-table-middle>.ant-table-content>.ant-table-scroll>.ant-table-body>table';
const schemaTableClassName = css({
  [`${antTable}>.ant-table-thead>tr>th.ant-table-expand-icon-th`]: {
    padding: `${spacingMedium}px 0`,
    width: '32px',
  },
  [`${antTable}>.ant-table-thead>tr>th.ant-table-row-cell-break-word`]: {
    padding: `${spacingMedium}px 0`,
  },
  [`${antTable}>.ant-table-tbody>tr>td.ant-table-row-cell-break-word`]: {
    padding: `${spacingMedium}px 0`,
  },
  [`${antTable}>.ant-table-tbody>tr.section-header-row>td.ant-table-row-cell-break-word`]: {
    padding: '0',
    backgroundColor: '#EEEEEE',
    width: '32px',
  },
  [`${antTable}>.ant-table-tbody>tr.section-header-row>td.ant-table-row-expand-icon-cell`]: {
    padding: '0',
    backgroundColor: '#EEEEEE',
  },
  '.outer-table .ant-table-body': {
    // !important to override inline style of overflowX: scroll
    overflowX: 'auto !important',
    overflowY: 'hidden',
  },
  '.inner-table .ant-table-body': {
    // !important to override inline style of overflowY: scroll
    overflowY: 'auto !important',
  },
  '.ant-table-expanded-row td': {
    backgroundColor: 'white',
  },
  '.inner-table': {
    maxWidth: 800,
  },
  '.outer-table': {
    maxWidth: 800,
  },
  '.primary-text': {
    color: gray800,
  },
  '.section-header-row': {
    lineHeight: '32px',
    cursor: 'pointer',
  },
});
