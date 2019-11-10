import React, { Component } from 'react';
import PropTypes from 'prop-types';
import "./HtmlTableView.css";
import Table from 'react-bootstrap/es/Table';
import Utils from '../utils/Utils';

class HtmlTableView extends Component {
  static propTypes = {
    columns: PropTypes.array.isRequired,
    // Array of objects like {key: <node>, value: <node>, title: string}
    values: PropTypes.array.isRequired,
    styles: PropTypes.object,
  };

  render() {
    const styles = this.props.styles || {};
    return (
      <Table hover className="HtmlTableView" style={styles['table']}>
        <tbody>
        <tr style={styles['tr']}>
          { this.props.columns.map((c, idx) => {
            let style;
            if (idx === 0) {
              style = styles['th-first'] || styles['th'];
            } else {
              style = styles['th'];
            }
            return <th key={idx} style={style}>{c}</th>;
          }
          )}
        </tr>
          { this.props.values.map((vObj, index) => {
              debugger;
              return <tr key={index} style={styles['tr']}>
                  <td key={vObj.key + "-key"} style={styles['td-first'] || styles['td']}>
                      {vObj.key}
                  </td>
                  <td title={vObj.valueTitle} key={vObj.key + "-value"}>
                      {vObj.value}
                  </td>

                  {/*{ Object.entries(vObj).map((idx, [key, value]) => {*/}
                  {/*  let style;*/}
                  {/*  if (idx === 0) {*/}
                  {/*    style = styles['td-first'] || styles['td'];*/}
                  {/*  } else {*/}
                  {/*    style = styles['td'];*/}
                  {/*  }*/}
                  {/*  return <td key={idx} style={style}>{v}</td>;*/}
                  {/*}*/}
                  {/*)}*/}
              </tr>
          })}
        </tbody>
      </Table>
    );
  }
}

export default HtmlTableView;
