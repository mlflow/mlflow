import {
  fetchMissingParents,
  getParentRunIdsToFetch,
  getParentRunTagName,
  searchRunsPayload,
} from './actions';
import { MlflowService } from './sdk/MlflowService';

const a = {
  info: { run_id: 'a' },
  data: {
    tags: [
      {
        key: getParentRunTagName(),
        value: 'aParent',
      },
    ],
  },
};
const b = {
  info: { run_id: 'b' },
  data: {
    tags: [
      {
        key: getParentRunTagName(),
        value: 'bParent',
      },
    ],
  },
};
const aParent = { info: { run_id: 'aParent' } };
const bParent = { info: { run_id: 'bParent' } };

beforeEach(() => {
  jest
    .spyOn(MlflowService, 'searchRuns')
    .mockImplementation(({ success }) => success({ runs: [a, b, aParent] }));

  jest
    .spyOn(MlflowService, 'getRun')
    .mockImplementation(({ data, success }) => success({ run: { info: { run_id: data.run_id } } }));
});

afterEach(() => {
  MlflowService.searchRuns.mockRestore();
  MlflowService.getRun.mockRestore();
});

describe('fetchMissingParents', () => {
  it('should not explode if no runs', () => {
    const res = { nextPageToken: 'something' };
    expect(fetchMissingParents(res)).toBe(res);
  });

  it('should return res if runs empty', () => {
    const res = {
      runs: [],
      nextPageToken: 'something',
    };
    expect(fetchMissingParents(res)).toEqual(res);
  });

  it('should merge received parent runs', () => {
    const res = { runs: [a, b] };
    return fetchMissingParents(res).then((runs) => {
      expect(runs).toEqual({ runs: [a, b, aParent, bParent] });
    });
  });

  it('should return given runs even if no parent runs', () => {
    const res = { runs: [a, b, aParent, bParent] };
    return fetchMissingParents(res).then((runs) => {
      expect(runs).toEqual({ runs: [a, b, aParent, bParent] });
    });
  });

  it('should handle deleted parent runs', () => {
    const mockParentRunDeletedError = {
      getErrorCode() {
        return 'RESOURCE_DOES_NOT_EXIST';
      },
    };

    jest.spyOn(MlflowService, 'getRun').mockImplementation(({ data, success }) => {
      if (data.run_id === 'aParent') {
        success({ run: { info: { run_id: data.run_id } } });
      } else {
        throw mockParentRunDeletedError;
      }
    });

    const res = { runs: [a, b] };
    return fetchMissingParents(res).then((runs) => {
      expect(runs).toEqual({ runs: [a, b, aParent] });
    });
  });

  it('should throw for unexpected exceptions encountered during run resolution', async () => {
    const mockUnexpectedGetRunError = {
      getErrorCode() {
        return 'INTERNAL_ERROR';
      },
    };

    jest.spyOn(MlflowService, 'getRun').mockImplementation(() => {
      throw mockUnexpectedGetRunError;
    });

    const res = { runs: [a, b] };
    await expect(fetchMissingParents(res)).rejects.toEqual(mockUnexpectedGetRunError);
  });
});

describe('getParentRunIdsToFetch', () => {
  it('should return empty array if no runs', () => {
    expect(getParentRunIdsToFetch([])).toEqual([]);
  });

  it('should return an array of absent parents ids given an array of runs', () => {
    expect(getParentRunIdsToFetch([a, b, bParent])).toEqual(['aParent']);
    expect(getParentRunIdsToFetch([a, b])).toEqual(['aParent', 'bParent']);
    expect(getParentRunIdsToFetch([a, b, aParent, bParent])).toEqual([]);
    expect(getParentRunIdsToFetch([a, bParent])).toEqual(['aParent']);
  });
});

describe('searchRunsPayload', () => {
  it('should fetch parents only if shouldFetchParents is true', async () => {
    await searchRunsPayload({}).then((res) => {
      expect(res).toEqual({ runs: [a, b, aParent] });
    });

    await searchRunsPayload({ shouldFetchParents: false }).then((res) => {
      expect(res).toEqual({ runs: [a, b, aParent] });
    });

    await searchRunsPayload({ shouldFetchParents: true }).then((res) => {
      expect(res).toEqual({ runs: [a, b, aParent, bParent] });
    });
  });
});
