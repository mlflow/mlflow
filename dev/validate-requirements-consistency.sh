#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REQUIREMENTS_DIR=$(realpath $SCRIPT_DIR/../requirements)

# Skinny
python $SCRIPT_DIR/generate_requirements.py --requirements-yaml-location $REQUIREMENTS_DIR/skinny-requirements.yaml --requirements-txt-output-location /tmp/skinny-requirements-for-validation.txt

if diff /tmp/skinny-requirements-for-validation.txt $REQUIREMENTS_DIR/skinny-requirements.txt ; then
echo "$REQUIREMENTS_DIR/skinny-requirements.txt is consistent with $REQUIREMENTS_DIR/skinny-requirements.yaml"
else
REGENERATE_CMD=$(cat <<-END
    python dev/generate_requirements.py \\
      --requirements-yaml-location requirements/skinny-requirements.yaml \\
      --requirements-txt-output-location requirements/skinny-requirements.txt
END
)
printf "\nskinny-requirements.txt is not consistent with skinny-requirements.yaml!\n\n"
printf "Run the following command from the root of the MLflow repository to regenerate skinny-requirements.txt:\n\n ${REGENERATE_CMD}\n"
exit 1
fi

# Core
python $SCRIPT_DIR/generate_requirements.py --requirements-yaml-location $REQUIREMENTS_DIR/core-requirements.yaml --requirements-txt-output-location /tmp/core-requirements-for-validation.txt

if diff /tmp/core-requirements-for-validation.txt $REQUIREMENTS_DIR/core-requirements.txt ; then
echo "$REQUIREMENTS_DIR/core-requirements.txt is consistent with $REQUIREMENTS_DIR/core-requirements.yaml"
else
REGENERATE_CMD=$(cat <<-END
    python dev/generate_requirements.py \\
      --requirements-yaml-location requirements/core-requirements.yaml \\
      --requirements-txt-output-location requirements/core-requirements.txt
END
)
printf "\ncore-requirements.txt is not consistent with core-requirements.yaml!\n\n"
printf "Run the following command from the root of the MLflow repository to regenerate core-requirements.txt:\n\n ${REGENERATE_CMD}\n"
exit 1
fi

# Gateway
python $SCRIPT_DIR/generate_requirements.py --requirements-yaml-location $REQUIREMENTS_DIR/gateway-requirements.yaml --requirements-txt-output-location /tmp/gateway-requirements-for-validation.txt

if diff /tmp/gateway-requirements-for-validation.txt $REQUIREMENTS_DIR/gateway-requirements.txt ; then
echo "$REQUIREMENTS_DIR/gateway-requirements.txt is consistent with $REQUIREMENTS_DIR/gateway-requirements.yaml"
else
REGENERATE_CMD=$(cat <<-END
    python dev/generate_requirements.py \\
      --requirements-yaml-location requirements/gateway-requirements.yaml \\
      --requirements-txt-output-location requirements/gateway-requirements.txt
END
)
printf "\ngateway-requirements.txt is not consistent with cgatewayore-requirements.yaml!\n\n"
printf "Run the following command from the root of the MLflow repository to regenerate gateway-requirements.txt:\n\n ${REGENERATE_CMD}\n"
exit 1
fi
