import {
  transformGetRunResponse,
  transformSearchRunsResponse,
  transformSearchExperimentsResponse,
} from './FieldNameTransformers';

describe('transformSearchRunsResponse', () => {
  it('transforms the search runs response', () => {
    const snakeCaseResponse = {
      runs: [
        {
          info: {
            experiment_id: '1',
            artifact_uri: '/something',
            run_name: 'Run abc',
            run_uuid: '123',
            status: 'ACTIVE',
          },
          inputs: {
            model_inputs: [
              {
                model_id: 'm-1',
              },
            ],
            dataset_inputs: [
              {
                dataset: {
                  name: 'dataset',
                  digest: '608a286f',
                  source_type: 'code',
                  source: '{}',
                  profile: '{}',
                },
              },
            ],
          },
          outputs: {
            model_outputs: [
              {
                model_id: 'm-2',
              },
            ],
          },
        },
        {
          info: {
            experiment_id: '2',
            artifact_uri: '/something_else',
            run_name: 'Run xyz',
            run_uuid: '124',
            status: 'ACTIVE',
          },
        },
      ],
    };

    const transformedResponse = transformSearchRunsResponse(snakeCaseResponse);

    expect(transformedResponse).toEqual({
      runs: [
        {
          info: expect.objectContaining({
            experimentId: '1',
            artifactUri: '/something',
            runName: 'Run abc',
            runUuid: '123',
            status: 'ACTIVE',
          }),
          inputs: expect.objectContaining({
            modelInputs: [
              expect.objectContaining({
                modelId: 'm-1',
              }),
            ],
            datasetInputs: [
              expect.objectContaining({
                dataset: expect.objectContaining({
                  name: 'dataset',
                }),
              }),
            ],
          }),
          outputs: expect.objectContaining({
            modelOutputs: [
              expect.objectContaining({
                modelId: 'm-2',
              }),
            ],
          }),
        },
        {
          info: expect.objectContaining({
            experimentId: '2',
            artifactUri: '/something_else',
            runName: 'Run xyz',
            runUuid: '124',
            status: 'ACTIVE',
          }),
        },
      ],
    });
  });

  it('returns the response if it is invalid', () => {
    const response = null;

    const transformedResponse = transformSearchRunsResponse(response);

    expect(transformedResponse).toBeNull();
  });
});

describe('transformGetRunResponse', () => {
  it('transforms the get run response', () => {
    const originalResponse = {
      run: {
        info: {
          experiment_id: '1',
          artifact_uri: '/something',
          run_name: 'Run abc',
          run_uuid: '123',
          status: 'ACTIVE',
        },
      },
    };

    const transformedResponse = transformGetRunResponse(originalResponse);

    expect(transformedResponse).toEqual({
      run: {
        info: expect.objectContaining({
          experimentId: '1',
          artifactUri: '/something',
          runName: 'Run abc',
          runUuid: '123',
          status: 'ACTIVE',
        }),
      },
    });
  });

  it('returns the response if it is invalid', () => {
    const originalResponse = null;

    const transformedResponse = transformGetRunResponse(originalResponse);

    expect(transformedResponse).toBeNull();
  });
});

describe('transformSearchExperimentsResponse', () => {
  it('transforms the search experiments response', () => {
    const originalResponse = {
      experiments: [
        {
          experiment_id: '1',
          name: 'Experiment 1',
          lifecycle_stage: 'active',
        },
        {
          experiment_id: '2',
          name: 'Experiment 2',
          lifecycle_stage: 'deleted',
        },
      ],
    };

    const transformedResponse = transformSearchExperimentsResponse(originalResponse);

    expect(transformedResponse).toEqual({
      experiments: [
        expect.objectContaining({
          experimentId: '1',
          name: 'Experiment 1',
          lifecycleStage: 'active',
        }),
        expect.objectContaining({
          experimentId: '2',
          name: 'Experiment 2',
          lifecycleStage: 'deleted',
        }),
      ],
    });
  });

  it('returns the response if it is invalid', () => {
    const originalResponse = null;

    const transformedResponse = transformSearchExperimentsResponse(originalResponse);

    expect(transformedResponse).toBeNull();
  });
});
