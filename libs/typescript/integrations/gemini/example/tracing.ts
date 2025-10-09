
import * as mlflow from 'mlflow-tracing';
import { GoogleGenAI } from '@google/genai';
import { tracedGemini } from 'mlflow-gemini';

async function main() {
  mlflow.init({
    trackingUri: 'http://localhost:5000',
    experimentId: '252237888791228129'
  });

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is not set.');
  }

  const client = tracedGemini(new GoogleGenAI({ apiKey }));

  const response = await client.models.generateContent({
    model: 'gemini-2.0-flash-001',
    contents: 'Why is the sky blue?'
  });

  console.log(response.text);

//   await mlflow.flushTraces();
}

main().catch((error) => {
  console.error('Error running Gemini tracing example:', error);
  process.exitCode = 1;
});