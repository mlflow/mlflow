import { test, jest, expect, describe } from '@jest/globals';
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

  // input value == undefined, no error message expected
  experimentNameValidator(undefined, undefined, mockCallback);
  expect(mockCallback).toHaveBeenCalledWith(undefined);
});

describe('modelNameValidator should work properly', () => {
  test('should invoke callback with undefined for empty name', () => {
    const mockCallback = jest.fn((err) => err);
    modelNameValidator(undefined, '', mockCallback);
    expect(mockCallback).toHaveBeenCalledWith(undefined);
  });

  test('should invoke callback with undefined for undefined name', () => {
    const mockCallback = jest.fn((err) => err);
    modelNameValidator(undefined, undefined, mockCallback);
    expect(mockCallback).toHaveBeenCalledWith(undefined);
  });

  test('should invoke callback with error message when model exists', async () => {
    // getRegisteredModel returns resolved promise indicates model exists
    ModelRegistryService.getRegisteredModel = jest.fn(() => Promise.resolve());
    const mockCallback = jest.fn((err) => err);
    const modelName = 'model A';
    modelNameValidator(undefined, modelName, mockCallback);
    // Wait for all microtasks (promise .then()/.catch() handlers) to complete
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockCallback).toHaveBeenCalledWith(`Model "${modelName}" already exists.`);
  });

  test('should invoke callback with undefined when model does not exist', async () => {
    // getRegisteredModel returns rejected promise indicates model does not exist
    ModelRegistryService.getRegisteredModel = jest.fn(() => Promise.reject());
    const mockCallback = jest.fn((err) => err);
    const modelName = 'model A';
    modelNameValidator(undefined, modelName, mockCallback);
    // Wait for all microtasks (promise .then()/.catch() handlers) to complete
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(mockCallback).toHaveBeenCalledWith(undefined);
  });
});
