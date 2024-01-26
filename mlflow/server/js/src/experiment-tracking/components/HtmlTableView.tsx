/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
        className="html-table-view"
        data-test-id={this.props.testId}
        dataSource={this.props.values}
        columns={this.props.columns}
        scroll={this.props.scroll}
        size="middle"
        pagination={false}
        style={styles}
      />
    );
  }
}
