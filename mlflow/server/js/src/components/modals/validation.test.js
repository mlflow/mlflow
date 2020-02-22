import { getExperimentNameValidator } from './validation';

test('ExperimentNameValidator works properly', () => {
  const experimentNames = ['Default', 'Test Experiment'];
  const value = experimentNames[0];
  const experimentNameValidator = getExperimentNameValidator(() => experimentNames);

  const mockCallback = jest.fn((err) => err);

  // pass one of the existing experiments as input value
  experimentNameValidator(undefined, value, mockCallback);
  expect(mockCallback).toHaveBeenCalledWith(`Experiment "${value}" already exists.`);

  // no input value passed, no error message expected
  experimentNameValidator(undefined, '', mockCallback);
  expect(mockCallback).toHaveBeenCalledWith(undefined);
});
