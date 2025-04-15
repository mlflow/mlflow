import { EvaluationTableParseError, fetchEvaluationTableArtifact } from './EvaluationArtifactService';

const mockGetArtifactChunkedText = jest.fn();

jest.mock('../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/ArtifactUtils')>('../../common/utils/ArtifactUtils'),
  getArtifactChunkedText: () => mockGetArtifactChunkedText(),
}));

describe('fetchEvaluationTableArtifact', () => {
  const MOCK_RESPONSE = {
    columns: ['inputs', 'outputs', 'targets'],
    data: [
      ['Input A', 'Output A', 'Prompt A'],
      ['Input B', 'Output B', 'Prompt B'],
      ['Input C', 'Output C', 'Prompt C'],
    ],
  };
  const mockResponse = (response: Partial<typeof MOCK_RESPONSE>) =>
    mockGetArtifactChunkedText.mockImplementation(() => Promise.resolve(JSON.stringify(response)));

  beforeEach(() => {
    mockResponse(MOCK_RESPONSE);
  });

  it('correctly fetches and parses data', async () => {
    const result = await fetchEvaluationTableArtifact('run_1', '/some/artifact');
    expect(result.entries).toHaveLength(3);
    expect(result.entries.map(({ inputs }) => inputs)).toEqual(['Input A', 'Input B', 'Input C']);
    expect(result.entries.map(({ outputs }) => outputs)).toEqual(['Output A', 'Output B', 'Output C']);
    expect(result.entries.map(({ targets }) => targets)).toEqual(['Prompt A', 'Prompt B', 'Prompt C']);
  });

  it('fails on malformed response without columns field', async () => {
    mockResponse({ data: [] });
    await expect(fetchEvaluationTableArtifact('run_1', '/some/artifact')).rejects.toThrow(
      /does not contain "columns" field/,
    );
  });

  it('fails on malformed response without data field', async () => {
    mockResponse({ columns: [] });
    await expect(fetchEvaluationTableArtifact('run_1', '/some/artifact')).rejects.toThrow(
      /does not contain "data" field/,
    );
  });

  it('fails with specific error on non-JSON response', async () => {
    mockGetArtifactChunkedText.mockImplementation(() => Promise.resolve('[[[[[some invalid json{{'));

    await expect(fetchEvaluationTableArtifact('run_1', '/some/artifact')).rejects.toThrow(EvaluationTableParseError);
  });
});
