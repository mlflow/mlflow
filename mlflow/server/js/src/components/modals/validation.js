import { string, object } from 'yup';

export const getValidationSchema = (type) => {
  const fieldType = type[0].toUpperCase() + type.slice(1).toLowerCase();
  return object().shape({
    [`${type}Name`]: string().required(`${fieldType} name is a required field`),
  });
};
