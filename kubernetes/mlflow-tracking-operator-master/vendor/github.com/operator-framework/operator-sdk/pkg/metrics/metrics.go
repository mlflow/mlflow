// Copyright 2018 The Operator-SDK Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"context"
	"fmt"

	"github.com/operator-framework/operator-sdk/pkg/k8sutil"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/rest"
	crclient "sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

var log = logf.Log.WithName("metrics")

// PrometheusPortName defines the port name used in kubernetes deployment and service resources
const PrometheusPortName = "metrics"

// ExposeMetricsPort creates a Kubernetes Service to expose the passed metrics port.
func ExposeMetricsPort(ctx context.Context, port int32) (*v1.Service, error) {
	// We do not need to check the validity of the port, as controller-runtime
	// would error out and we would never get to this stage.
	s, err := initOperatorService(port, PrometheusPortName)
	if err != nil {
		if err == k8sutil.ErrNoNamespace {
			log.Info("Skipping metrics Service creation; not running in a cluster.")
			return nil, nil
		}
		return nil, fmt.Errorf("failed to initialize service object for metrics: %v", err)
	}
	service, err := createService(ctx, s)
	if err != nil {
		return nil, fmt.Errorf("failed to create or get service for metrics: %v", err)
	}

	return service, nil
}

func createService(ctx context.Context, s *v1.Service) (*v1.Service, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, err
	}

	client, err := crclient.New(config, crclient.Options{})
	if err != nil {
		return nil, err
	}

	if err := client.Create(ctx, s); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, err
		}
		// Get existing Service and return it
		existingService := &v1.Service{}
		err := client.Get(ctx, types.NamespacedName{
			Name:      s.Name,
			Namespace: s.Namespace,
		}, existingService)
		if err != nil {
			return nil, err
		}
		log.Info("Metrics Service object already exists", "name", existingService.Name)
		return existingService, nil
	}

	log.Info("Metrics Service object created", "name", s.Name)
	return s, nil
}

// initOperatorService returns the static service which exposes specifed port.
func initOperatorService(port int32, portName string) (*v1.Service, error) {
	operatorName, err := k8sutil.GetOperatorName()
	if err != nil {
		return nil, err
	}
	namespace, err := k8sutil.GetOperatorNamespace()
	if err != nil {
		return nil, err
	}
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      operatorName,
			Namespace: namespace,
			Labels:    map[string]string{"name": operatorName},
		},
		TypeMeta: metav1.TypeMeta{
			Kind:       "Service",
			APIVersion: "v1",
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{
					Port:     port,
					Protocol: v1.ProtocolTCP,
					TargetPort: intstr.IntOrString{
						Type:   intstr.Int,
						IntVal: port,
					},
					Name: portName,
				},
			},
			Selector: map[string]string{"name": operatorName},
		},
	}
	return service, nil
}
