import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './HtmlTableView.css';
import { Table } from 'antd';

export class HtmlTableView extends Component {
  static propTypes = {
    columns: PropTypes.array.isRequired,
    values: PropTypes.array.isRequired,
    styles: PropTypes.object,
    testId: PropTypes.string,
    scroll: PropTypes.object,
  };

  render() {
    const styles = this.props.styles || {};

    return (
      <Table
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
