import React, { useState } from 'react';

export const EvaluationMenu = ({ runUuid }: { runUuid: string }) => {
  const [visible, setVisible] = useState(true);

  return (
    <div style={{ marginTop: 16 }}>
      <button onClick={() => setVisible(!visible)}>
        Toggle Evaluation Visibility
      </button>

      {visible && (
        <div style={{ padding: 8, border: '1px solid #ccc', marginTop: 8 }}>
          <p>Evaluation options for run: {runUuid}</p>
        </div>
      )}
    </div>
  );
};

export default EvaluationMenu;
