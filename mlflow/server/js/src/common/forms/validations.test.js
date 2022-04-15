import { getExperimentNameValidator, modelNameValidator } from './validations';
import { Services as ModelRegistryService } from '../../model-registry/services';

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

describe('modelNameValidator should work properly', () => {
  test('should invoke callback with undefined for empty name', () => {
    const mockCallback = jest.fn((err) => err);
    modelNameValidator(undefined, '', mockCallback);
    expect(mockCallback).toHaveBeenCalledWith(undefined);
  });

  test('should invoke callback with error message when model exists', (done) => {
    // getRegisteredModel returns resolved promise indicates model exists
    ModelRegistryService.getRegisteredModel = jest.fn(() => Promise.resolve());
    const mockCallback = jest.fn((err) => err);
    const modelName = 'model A';
    modelNameValidator(undefined, modelName, mockCallback);
    // Check callback invocation in the next tick. We are doing this because returning a promise
    // in callback based validator leads to incorrect form error message behavior.
    setTimeout(() => {
      expect(mockCallback).toHaveBeenCalledWith(`Model "${modelName}" already exists.`);
      done();
    });
  });

  test('should invoke callback with undefined when model does not exist', (done) => {
    // getRegisteredModel returns rejected promise indicates model does not exist
    ModelRegistryService.getRegisteredModel = jest.fn(() => Promise.reject());
    const mockCallback = jest.fn((err) => err);
    const modelName = 'model A';
    modelNameValidator(undefined, modelName, mockCallback);
    // Check callback invocation in the next tick. We are doing this because returning a promise
    // in callback based validator leads to incorrect form error message behavior.
    setTimeout(() => {
      expect(mockCallback).toHaveBeenCalledWith(undefined);
      done();
    });
  });
});
