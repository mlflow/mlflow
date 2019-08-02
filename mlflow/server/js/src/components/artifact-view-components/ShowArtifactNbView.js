import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getRequestHeaders } from '../../setupAjaxHeaders';
import { parseNotebook, fromJS } from "@nteract/commutable";
import {
    Cell, Input, Prompt, Source, Outputs, Cells
} from "@nteract/presentational-components";
import {
    Output, StreamText, DisplayData, Media, KernelOutputError, ExecuteResult
} from "@nteract/outputs";
import Markdown from "@nteract/markdown";
import { Provider as MathJaxProvider } from "@nteract/mathjax";
import './ShowArtifactNbView.css';


class ShowArtifactNbView extends Component {
  constructor(props) {
    super(props);
    this.fetchArtifacts = this.fetchArtifacts.bind(this);
  }

  static propTypes = {
    runUuid: PropTypes.string.isRequired,
    path: PropTypes.string.isRequired,
  };

  state = {
    loading: true,
    error: undefined,
    immutableNotebook: undefined
  };

  componentDidMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    const { error, loading, immutableNotebook } = this.state;
    if (error) {
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else if (loading) {
      return (
        <div>
          Loading...
        </div>
      );
    } else if (immutableNotebook) {
      const language = immutableNotebook.getIn(
        ["metadata", "language_info", "codemirror_mode", "name"],
        immutableNotebook.getIn(
          ["metadata", "language_info", "codemirror_mode"],
          immutableNotebook.getIn(["metadata", "language_info", "name"], "text")
        )
      );

      const cellOrder = immutableNotebook.get("cellOrder");
      const cellMap = immutableNotebook.get("cellMap");

      return (
        <div className="notebook-container">
            <MathJaxProvider>
                <div className="notebook-preview">
                    <Cells>
                        {cellOrder.map(cellID => {
                          const cell = cellMap.get(cellID);
                          const cellType = cell.get("cell_type");
                          const source = cell.get("source");
                          const outputs = cell.get("outputs");

                          switch (cellType) {
                            case "code":
                              return (
                                <Cell key={cellID}>
                                  <Input>
                                    <Prompt />
                                    <Source language={language}>
                                      {source}
                                    </Source>
                                  </Input>
                                  <Outputs>
                                    {outputs ?
                                        outputs.map((output, index) => (
                                            <Output output={output} key={index}>
                                              <StreamText />
                                              <ExecuteResult>
                                                <Media.Image mediaType="image/jpeg"/>
                                                <Media.Image mediaType="image/gif"/>
                                                <Media.Image mediaType="image/png"/>
                                                <Media.HTML />
                                                <Media.Json />
                                                <Media.JavaScript />
                                                <Media.Plain />
                                              </ExecuteResult>
                                              <DisplayData>
                                                <Media.Image mediaType="image/jpeg"/>
                                                <Media.Image mediaType="image/gif"/>
                                                <Media.Image mediaType="image/png"/>
                                                <Media.HTML />
                                                <Media.Json />
                                                <Media.JavaScript />
                                                <Media.Plain />
                                              </DisplayData>
                                              <KernelOutputError />
                                            </Output>
                                        ))
                                        : null
                                    }
                                  </Outputs>
                                </Cell>
                              );
                            case "markdown":
                              return (
                                <Cell key={cellID}>
                                  <div className="content-margin">
                                    <Markdown source={source} />
                                  </div>
                                </Cell>
                              );
                            case "raw":
                              return (
                                <Cell key={cellID}>
                                  <pre className="raw-cell">
                                    {source}
                                  </pre>
                                </Cell>
                              );

                            default:
                              return (
                                <Cell key={cellID}>
                                  <Outputs>
                                    <pre>{`Cell Type "${cellType}" is not implemented`}</pre>
                                  </Outputs>
                                </Cell>
                              );
                          }
                        })}
                    </Cells>
                </div>
            </MathJaxProvider>
        </div>
      );
    } else {
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    }
  }

  fetchArtifacts() {
    const getArtifactRequest = new Request(getSrc(this.props.path, this.props.runUuid), {
      method: 'GET',
      redirect: 'follow',
      headers: new Headers(getRequestHeaders(document.cookie))
    });
    fetch(getArtifactRequest).then((response) => {
      return response.blob();
    }).then((blob) => {
      const fileReader = new FileReader();
      fileReader.onload = (event) => {
        const notebook = parseNotebook(event.target.result);
        const immutableNotebook = fromJS(notebook);
        this.setState({ immutableNotebook: immutableNotebook, loading: false, error: undefined });
      };
      fileReader.readAsText(blob);
    }).catch(error => this.setState({ error, loading: false, immutableNotebook: undefined }));
  }
}

export default ShowArtifactNbView;
