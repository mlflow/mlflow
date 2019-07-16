import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import './ShowArtifactNbView.css';
import { getRequestHeaders } from '../../setupAjaxHeaders';
import { parseNotebook, fromJS } from "@nteract/commutable";
import {
    LightTheme, Cell, Input, Prompt, Source, Outputs, Cells
} from "@nteract/presentational-components";
import {
    Output, StreamText, DisplayData, Media, KernelOutputError, ExecuteResult
} from "@nteract/outputs";
import Markdown from "@nteract/markdown";
import { Provider as MathJaxProvider } from "@nteract/mathjax";


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

  componentWillMount() {
    this.fetchArtifacts();
  }

  componentDidUpdate(prevProps) {
    if (this.props.path !== prevProps.path || this.props.runUuid !== prevProps.runUuid) {
      this.fetchArtifacts();
    }
  }

  render() {
    if (this.state.loading) {
      return (
        <div>
          Loading...
        </div>
      );
    }
    if (this.state.error) {
      return (
        <div>
          Oops we couldn't load your file because of an error.
        </div>
      );
    } else {
      const notebook = this.state.immutableNotebook;

      const language = notebook.getIn(
        ["metadata", "language_info", "codemirror_mode", "name"],
        notebook.getIn(
          ["metadata", "language_info", "codemirror_mode"],
          notebook.getIn(["metadata", "language_info", "name"], "text")
        )
      );

      const cellOrder = notebook.get("cellOrder");
      const cellMap = notebook.get("cellMap");

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
                                    <Source language={language} theme="light">
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
                                  <style jsx>{`
                                    .content-margin {
                                      padding-left: calc(var(--prompt-width, 50px) + 10px);
                                      padding-top: 10px;
                                      padding-bottom: 10px;
                                      padding-right: 10px;
                                    }
                                  `}</style>
                                </Cell>
                              );
                            case "raw":
                              return (
                                <Cell key={cellID}>
                                  <pre className="raw-cell">
                                    {source}
                                    <style jsx>{`
                                      raw-cell {
                                        background: repeating-linear-gradient(
                                          -45deg,
                                          transparent,
                                          transparent 10px,
                                          #efefef 10px,
                                          #f1f1f1 20px
                                        );
                                      }
                                    `}</style>
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

                    <style>{`:root {
                      ${LightTheme}
                        --theme-cell-shadow-hover: none;
                        --theme-cell-shadow-focus: none;
                        --theme-cell-prompt-bg-hover: var(--theme-cell-prompt-bg);
                        --theme-cell-prompt-bg-focus: var(--theme-cell-prompt-bg);
                        --theme-cell-prompt-fg-hover: var(--theme-cell-prompt-fg);
                        --theme-cell-prompt-fg-focus: var(--theme-cell-prompt-fg);
                      }
                    `}</style>
                </div>
            </MathJaxProvider>
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
        this.setState({ immutableNotebook: immutableNotebook, loading: false });
      };
      fileReader.readAsText(blob);
    });
  }
}

export default ShowArtifactNbView;
