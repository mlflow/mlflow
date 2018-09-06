# Builds the MLflow Javadoc and places it into build/html/java_api/

set -ex
pushd ../mlflow/java/client/
mvn javadoc:javadoc
popd
mkdir -p build/html/java_api/
cp -r ../mlflow/java/client/target/site/apidocs/* build/html/java_api/
echo "Copied JavaDoc into docs/build/html/java_api/"
