import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { getSrc } from './ShowArtifactPage';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';
import { Table } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import Papa from 'papaparse';

const ShowArtifactTableView = ({ runUuid, path, getArtifact }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState();
  const [data, setData] = useState();
  const [headers, setHeaders] = useState();
  const [text, setText] = useState();

  const MAX_NUM_ROWS = 500;

  useEffect(() => {
    resetState();
    fetchArtifacts({ path, runUuid, getArtifact });
  }, [runUuid, path, getArtifact]);

  function resetState() {
    setError();
    setData();
    setHeaders();
    setText();
    setLoading(true);
  }

  function fetchArtifacts(artifactData) {
    const artifactLocation = getSrc(artifactData.path, artifactData.runUuid);
    artifactData
      .getArtifact(artifactLocation)
      .then((artifactText) => {
        try {
          const result = Papa.parse(artifactText, {
            header: true,
            preview: MAX_NUM_ROWS,
            skipEmptyLines: 'greedy',
          });
          const dataPreview = result.data;

          if (result.errors.length > 0) {
            throw Error(result.errors[0].message);
          }

          setLoading(false);
          setHeaders(result.meta.fields);
          setData(dataPreview);
        } catch (_) {
          setLoading(false);
          setText(artifactText);
        }
      })
      .catch((e) => {
        setError(e);
        setLoading(false);
      });
  }

  if (loading) {
    return <div className='artifact-text-view-loading'>Loading...</div>;
  }
  if (error) {
    return (
      <div className='artifact-text-view-error'>
        Oops we couldn't load your file because of an error.
      </div>
    );
  }

  if (data) {
    const columns = headers.map((f) => ({
      title: f,
      dataIndex: f,
      key: f,
      sorter: (a, b) => a[f].localeCompare(b[f]),
      width: 200,
      ellipsis: {
        showTitle: true,
      },
    }));

    const numRows = data.length;

    return (
      <div css={{ overscrollBehaviorX: 'contain', overflowX: 'scroll', margin: 10 }}>
        <span css={{ display: 'flex', justifyContent: 'center' }}>
          <FormattedMessage
            defaultMessage='Previewing the first {numRows} rows'
            description='Title for showing the number of rows in the parsed data preview'
            values={{ numRows }}
          />
        </span>
        <Table
          columns={columns}
          dataSource={data}
          pagination={false}
          sticky
          scroll={{ x: 'min-content', y: true }}
        />
      </div>
    );
  } else {
    return (
      <div className='ShowArtifactPage'>
        <div className='text-area-border-box'>{text}</div>
      </div>
    );
  }
};

ShowArtifactTableView.propTypes = {
  runUuid: PropTypes.string.isRequired,
  path: PropTypes.string.isRequired,
  getArtifact: PropTypes.func,
};

ShowArtifactTableView.defaultProps = {
  getArtifact: getArtifactContent,
};

export default ShowArtifactTableView;
