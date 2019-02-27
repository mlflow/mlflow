// Copyright 2019 The Operator-SDK Authors
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

package scorecard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"strings"
	"time"

	"github.com/operator-framework/operator-sdk/internal/util/fileutil"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// checkSpecAndStat checks that the spec and status blocks exist. If noStore is set to true, this function
// will not store the result of the test in scTests and will instead just wait until the spec and
// status blocks exist or return an error after the timeout.
func checkSpecAndStat(runtimeClient client.Client, obj *unstructured.Unstructured, noStore bool) error {
	testSpec := scorecardTest{testType: basicOperator, name: "Spec Block Exists", maximumPoints: 1}
	testStat := scorecardTest{testType: basicOperator, name: "Status Block Exist", maximumPoints: 1}
	err := wait.Poll(time.Second*1, time.Second*time.Duration(viper.GetInt64(InitTimeoutOpt)), func() (bool, error) {
		err := runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
		if err != nil {
			return false, fmt.Errorf("error getting custom resource: %v", err)
		}
		var specPass, statusPass bool
		if obj.Object["spec"] != nil {
			testSpec.earnedPoints = 1
			specPass = true
		}

		if obj.Object["status"] != nil {
			testStat.earnedPoints = 1
			statusPass = true
		}
		return statusPass && specPass, nil
	})
	if !noStore {
		scTests = append(scTests, testSpec, testStat)
	}
	if err != nil && err != wait.ErrWaitTimeout {
		return err
	}
	if testSpec.earnedPoints != 1 {
		scSuggestions = append(scSuggestions, "Add a 'spec' field to your Custom Resource")
	}
	if testStat.earnedPoints != 1 {
		scSuggestions = append(scSuggestions, "Add a 'status' field to your Custom Resource")
	}
	return nil
}

// TODO: user specified tests for operators

// checkStatusUpdate looks at all fields in the spec section of a custom resource and attempts to modify them and
// see if the status changes as a result. This is a bit prone to breakage as this is a black box test and we don't
// know much about how the operators we are testing actually work and may pass an invalid value. In the future, we
// should use user-specified tests
func checkStatusUpdate(runtimeClient client.Client, obj *unstructured.Unstructured) error {
	test := scorecardTest{testType: basicOperator, name: "Operator actions are reflected in status", maximumPoints: 1}
	err := runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
	if err != nil {
		return fmt.Errorf("error getting custom resource: %v", err)
	}
	if obj.Object["status"] == nil || obj.Object["spec"] == nil {
		scTests = append(scTests, test)
		return nil
	}
	statCopy := make(map[string]interface{})
	for k, v := range obj.Object["status"].(map[string]interface{}) {
		statCopy[k] = v
	}
	specMap := obj.Object["spec"].(map[string]interface{})
	err = modifySpecAndCheck(specMap, obj)
	if err != nil {
		test.earnedPoints = 0
		scSuggestions = append(scSuggestions, "Make sure that the 'status' block is always updated to reflect changes after the 'spec' block is changed")
		scTests = append(scTests, test)
		return nil
	}
	test.earnedPoints = 1
	scTests = append(scTests, test)
	return nil
}

// modifySpecAndCheck is a helper function for checkStatusUpdate
func modifySpecAndCheck(specMap map[string]interface{}, obj *unstructured.Unstructured) error {
	statCopy := make(map[string]interface{})
	for k, v := range obj.Object["status"].(map[string]interface{}) {
		statCopy[k] = v
	}
	var err error
	for k, v := range specMap {
		mapType := false
		switch t := v.(type) {
		case int64:
			specMap[k] = specMap[k].(int64) + 1
		case float64:
			specMap[k] = specMap[k].(float64) + 1
		case string:
			// TODO: try and find out how to make this better
			// Since strings may be very operator specific, this test may not work correctly in many cases
			specMap[k] = fmt.Sprintf("operator sdk test value %f", rand.Float64())
		case bool:
			specMap[k] = !specMap[k].(bool)
		case map[string]interface{}:
			mapType = true
			err = modifySpecAndCheck(specMap[k].(map[string]interface{}), obj)
		case []map[string]interface{}:
			mapType = true
			for _, item := range specMap[k].([]map[string]interface{}) {
				err = modifySpecAndCheck(item, obj)
				if err != nil {
					break
				}
			}
		case []interface{}: // TODO: Decide how this should be handled
		default:
			fmt.Printf("Unknown type for key (%s) in spec: (%v)\n", k, reflect.TypeOf(t))
		}
		if !mapType {
			if err := runtimeClient.Update(context.TODO(), obj); err != nil {
				return fmt.Errorf("failed to update object: %v", err)
			}
			err = wait.Poll(time.Second*1, time.Second*15, func() (done bool, err error) {
				err = runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
				if err != nil {
					return false, err
				}
				return !reflect.DeepEqual(statCopy, obj.Object["status"]), nil
			})
		}
		if err != nil {
			return err
		}
		//reset stat copy to match
		statCopy = make(map[string]interface{})
		for k, v := range obj.Object["status"].(map[string]interface{}) {
			statCopy[k] = v
		}
	}
	return nil
}

// wiritingIntoCRsHasEffect simply looks at the proxy logs and verifies that the operator is sending PUT
// and/or POST requests to the API server, which should mean that it is creating or modifying resources.
func writingIntoCRsHasEffect(obj *unstructured.Unstructured) (string, error) {
	test := scorecardTest{testType: basicOperator, name: "Writing into CRs has an effect", maximumPoints: 1}
	kubeclient, err := kubernetes.NewForConfig(kubeconfig)
	if err != nil {
		return "", fmt.Errorf("failed to create kubeclient: %v", err)
	}
	dep := &appsv1.Deployment{}
	err = runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: deploymentName}, dep)
	if err != nil {
		return "", fmt.Errorf("failed to get newly created operator deployment: %v", err)
	}
	set := labels.Set(dep.Spec.Selector.MatchLabels)
	pods := &v1.PodList{}
	err = runtimeClient.List(context.TODO(), &client.ListOptions{LabelSelector: set.AsSelector()}, pods)
	if err != nil {
		return "", fmt.Errorf("failed to get list of pods in deployment: %v", err)
	}
	proxyPod = &pods.Items[0]
	req := kubeclient.CoreV1().Pods(obj.GetNamespace()).GetLogs(proxyPod.GetName(), &v1.PodLogOptions{Container: "scorecard-proxy"})
	readCloser, err := req.Stream()
	if err != nil {
		return "", fmt.Errorf("failed to get logs: %v", err)
	}
	defer func() {
		if err := readCloser.Close(); err != nil && !fileutil.IsClosedError(err) {
			log.Errorf("Failed to close pod log reader: (%v)", err)
		}
	}()
	buf := new(bytes.Buffer)
	_, err = buf.ReadFrom(readCloser)
	if err != nil {
		return "", fmt.Errorf("test failed and failed to read pod logs: %v", err)
	}
	logs := buf.String()
	msgMap := make(map[string]interface{})
	for _, msg := range strings.Split(logs, "\n") {
		if err := json.Unmarshal([]byte(msg), &msgMap); err != nil {
			continue
		}
		method, ok := msgMap["method"].(string)
		if !ok {
			continue
		}
		if method == "PUT" || method == "POST" {
			test.earnedPoints = 1
			break
		}
	}
	scTests = append(scTests, test)
	if test.earnedPoints != 1 {
		scSuggestions = append(scSuggestions, "The operator should write into objects to update state. No PUT or POST requests from you operator were recorded by the scorecard.")
	}
	return buf.String(), nil
}
