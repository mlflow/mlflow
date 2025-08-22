import { useEffect, useState } from 'react';

import { getTraceAttachment } from '@databricks/web-shared/model-trace-explorer';

export const ModelTraceExplorerAttachmentRenderer = ({
  title,
  id,
  traceId,
  contentType,
}: {
  title: string;
  id: string;
  traceId: string;
  contentType: string;
}) => {
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  console.log(id, traceId, contentType);

  useEffect(() => {
    if (id && contentType) {
      getTraceAttachment(traceId, id).then((data) => {
        if (data) {
          const url = URL.createObjectURL(new Blob([new Uint8Array(data)], { type: contentType }));
          setMediaUrl(url);
        }
      });
    }
  }, [id, contentType, traceId]);

  if (mediaUrl === null) {
    return <div>Loading ...</div>;
  }

  const isAudio = contentType.startsWith('audio/');
  const isImage = contentType.startsWith('image/');
  const isPdf = contentType === 'application/pdf';

  if (isAudio) {
    return (
      <div>
        <div>{title}</div>
        <br />
        <audio controls>
          <source src={mediaUrl} type={contentType} />
          Your browser does not support the audio element.
        </audio>
      </div>
    );
  }

  if (isImage) {
    return (
      <div>
        <div>{title}</div>
        <br />
        <img src={mediaUrl} alt={`Attachment ${id}`} />
      </div>
    );
  }

  if (isPdf) {
    return (
      <div>
        <div>{title}</div>
        <br />
        <iframe
          src={mediaUrl}
          width="100%"
          height="600px"
          title={`PDF Attachment ${id}`}
          style={{ border: '1px solid #ccc' }}
        />
      </div>
    );
  }

  return (
    <div>
      {mediaUrl && (
        <a href={mediaUrl} download={`attachment-${id}`}>
          Download {contentType} attachment
        </a>
      )}
    </div>
  );
};
