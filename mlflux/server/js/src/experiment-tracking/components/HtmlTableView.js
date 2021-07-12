import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './HtmlTableView.css';
import { Table } from 'react-bootstrap';

export class HtmlTableView extends Component {
  static propTypes = {
    columns: PropTypes.array.isRequired,
    values: PropTypes.array.isRequired,
    styles: PropTypes.object,
  };

  render() {
    const styles = this.props.styles || {};
    const { ...tableProps } = this.props;
    return (
      <Table hover className='HtmlTableView' style={styles['table']} {...tableProps}>
        <tbody>
          <tr style={styles['tr']}>
            {this.props.columns.map((c, idx) => {
              let style;
              if (idx === 0) {
                style = styles['th-first'] || styles['th'];
              } else {
                style = styles['th'];
              }
              return (
                <th key={idx} style={style}>
                  {c}
                </th>
              );
            })}
          </tr>
          {this.props.values.map((vArray, index) => (
            <tr key={index} style={styles['tr']}>
              {vArray.map((v, idx) => {
                let style;
                if (idx === 0) {
                  style = styles['td-first'] || styles['td'];
                } else {
                  style = styles['td'];
                }
                return (
                  <td key={idx} style={style}>
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </Table>
    );
  }
}
