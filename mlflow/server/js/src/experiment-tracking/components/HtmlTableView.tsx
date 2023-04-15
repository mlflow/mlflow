import React, { Component } from 'react';
import { LegacyTable } from '@databricks/design-system';
import './HtmlTableView.css';

type Props = {
  columns: any[];
  values: any[];
  styles?: any;
  testId?: string;
  scroll?: any;
};

export class HtmlTableView extends Component<Props> {
  render() {
    const styles = this.props.styles || {};

    return (
      <LegacyTable
        className='html-table-view'
        data-test-id={this.props.testId}
        dataSource={this.props.values}
        columns={this.props.columns}
        scroll={this.props.scroll}
        size='middle'
        pagination={false}
        style={styles}
      />
    );
  }
}
