import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

interface OTelIntegrationCardProps {
  name: string;
  logoPath: string;
}

export const OTelIntegrationCard: React.FC<OTelIntegrationCardProps> = ({ name, logoPath }) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      padding: '1rem',
      backgroundColor: 'var(--ifm-color-emphasis-100)',
      borderRadius: '8px',
      marginBottom: '1.5rem',
    }}
  >
    <img src={useBaseUrl(logoPath)} alt={`${name} Logo`} style={{ height: '36px' }} />
    <div>
      <h4>Integration via OpenTelemetry</h4>
      <p style={{ margin: 0 }}>
        {name} can be integrated with MLflow via OpenTelemetry. Configure {name}'s OpenTelemetry exporter to send traces
        to MLflow's OTLP endpoint.
      </p>
    </div>
  </div>
);

export default OTelIntegrationCard;
