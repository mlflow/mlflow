import React from 'react';
import { Table } from 'antd';
import PropType from 'prop-types';
import { css } from 'emotion';
import { gray800 } from '../../common/styles/color';
import { spacingMedium } from '../../common/styles/spacing';

const { Column } = Table;

export class SchemaTable extends React.PureComponent {
  static propTypes = {
    schema: PropType.object,
    defaultExpandAllRows: PropType.bool,
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

  getSchemaRowData = (schemaData) => {
    const rowData = [];
    schemaData.forEach((row, index) => {
      rowData[index] = {
        key: index,
        name: row.name ? row.name : '-',
        type: row.type ? row.type : '-',
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
            name: `Inputs (${schema.inputs.length})`,
            type: '',
            table: this.renderSchemaTable(schema.inputs, 'inputs'),
          },
          {
            key: '2',
            name: `Outputs (${schema.outputs.length})`,
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
          locale={{ emptyText: `No Schema.` }}
          dataSource={sectionHeaders}
          scroll={{ x: 240 }}
        >
          <Column
            key={1}
            title='Name'
            width='50%'
            dataIndex='name'
            render={this.renderSectionHeader}
          />
          <Column
            key={2}
            title='Type'
            width='50%'
            dataIndex='type'
            render={this.renderSectionHeader}
          />
        </Table>
      </div>
    );
  }
}

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
