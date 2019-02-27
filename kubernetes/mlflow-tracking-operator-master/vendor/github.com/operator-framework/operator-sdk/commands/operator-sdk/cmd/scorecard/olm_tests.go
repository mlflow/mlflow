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
	"context"
	"fmt"
	"strings"

	"github.com/operator-framework/operator-sdk/internal/util/k8sutil"

	olmapiv1alpha1 "github.com/operator-framework/operator-lifecycle-manager/pkg/api/apis/operators/v1alpha1"
	log "github.com/sirupsen/logrus"
	apiextv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func matchKind(kind1, kind2 string) bool {
	singularKind1, err := restMapper.ResourceSingularizer(kind1)
	if err != nil {
		singularKind1 = kind1
		log.Warningf("could not find singular version of %s", kind1)
	}
	singularKind2, err := restMapper.ResourceSingularizer(kind2)
	if err != nil {
		singularKind2 = kind2
		log.Warningf("could not find singular version of %s", kind2)
	}
	return strings.EqualFold(singularKind1, singularKind2)
}

// matchVersion checks if a CRD contains a specified version in a case insensitive manner
func matchVersion(version string, crd *apiextv1beta1.CustomResourceDefinition) bool {
	if strings.EqualFold(version, crd.Spec.Version) {
		return true
	}
	// crd.Spec.Version is deprecated, so check in crd.Spec.Versions as well
	for _, currVer := range crd.Spec.Versions {
		if strings.EqualFold(version, currVer.Name) {
			return true
		}
	}
	return false
}

// crdsHaveValidation makes sure that all CRDs have a validation block
func crdsHaveValidation(crdsDir string, runtimeClient client.Client, obj *unstructured.Unstructured) error {
	test := scorecardTest{testType: olmIntegration, name: "Provided APIs have validation"}
	crds, err := k8sutil.GetCRDs(crdsDir)
	if err != nil {
		return fmt.Errorf("failed to get CRDs in %s directory: %v", crdsDir, err)
	}
	err = runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
	if err != nil {
		return err
	}
	// TODO: we need to make this handle multiple CRs better/correctly
	for _, crd := range crds {
		test.maximumPoints++
		if crd.Spec.Validation == nil {
			scSuggestions = append(scSuggestions, fmt.Sprintf("Add CRD validation for %s/%s", crd.Spec.Names.Kind, crd.Spec.Version))
			continue
		}
		// check if the CRD matches the testing CR
		gvk := obj.GroupVersionKind()
		// Only check the validation block if the CRD and CR have the same Kind and Version
		if !(matchVersion(gvk.Version, crd) && matchKind(gvk.Kind, crd.Spec.Names.Kind)) {
			test.earnedPoints++
			continue
		}
		failed := false
		if obj.Object["spec"] != nil {
			spec := obj.Object["spec"].(map[string]interface{})
			for key := range spec {
				if _, ok := crd.Spec.Validation.OpenAPIV3Schema.Properties["spec"].Properties[key]; !ok {
					failed = true
					scSuggestions = append(scSuggestions, fmt.Sprintf("Add CRD validation for spec field `%s` in %s/%s", key, gvk.Kind, gvk.Version))
				}
			}
		}
		if obj.Object["status"] != nil {
			status := obj.Object["status"].(map[string]interface{})
			for key := range status {
				if _, ok := crd.Spec.Validation.OpenAPIV3Schema.Properties["status"].Properties[key]; !ok {
					failed = true
					scSuggestions = append(scSuggestions, fmt.Sprintf("Add CRD validation for status field `%s` in %s/%s", key, gvk.Kind, gvk.Version))
				}
			}
		}
		if !failed {
			test.earnedPoints++
		}
	}
	scTests = append(scTests, test)
	return nil
}

// crdsHaveResources checks to make sure that all owned CRDs have resources listed
func crdsHaveResources(csv *olmapiv1alpha1.ClusterServiceVersion) {
	test := scorecardTest{testType: olmIntegration, name: "Owned CRDs have resources listed"}
	for _, crd := range csv.Spec.CustomResourceDefinitions.Owned {
		test.maximumPoints++
		if len(crd.Resources) > 0 {
			test.earnedPoints++
		}
	}
	scTests = append(scTests, test)
	if test.earnedPoints == 0 {
		scSuggestions = append(scSuggestions, "Add resources to owned CRDs")
	}
}

// annotationsContainExamples makes sure that the CSVs list at least 1 example for the CR
func annotationsContainExamples(csv *olmapiv1alpha1.ClusterServiceVersion) {
	test := scorecardTest{testType: olmIntegration, name: "CRs have at least 1 example", maximumPoints: 1}
	if csv.Annotations != nil && csv.Annotations["alm-examples"] != "" {
		test.earnedPoints = 1
	}
	scTests = append(scTests, test)
	if test.earnedPoints == 0 {
		scSuggestions = append(scSuggestions, "Add an alm-examples annotation to your CSV to pass the "+test.name+" test")
	}
}

// statusDescriptors makes sure that all status fields found in the created CR has a matching descriptor in the CSV
func statusDescriptors(csv *olmapiv1alpha1.ClusterServiceVersion, runtimeClient client.Client, obj *unstructured.Unstructured) error {
	test := scorecardTest{testType: olmIntegration, name: "Status fields with descriptors"}
	err := runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
	if err != nil {
		return err
	}
	if obj.Object["status"] == nil {
		// what should we do if there is no status block? Maybe some kind of N/A type output?
		scTests = append(scTests, test)
		return nil
	}
	statusBlock := obj.Object["status"].(map[string]interface{})
	test.maximumPoints = len(statusBlock)
	var crd *olmapiv1alpha1.CRDDescription
	for _, owned := range csv.Spec.CustomResourceDefinitions.Owned {
		if owned.Kind == obj.GetKind() {
			crd = &owned
			break
		}
	}
	if crd == nil {
		scTests = append(scTests, test)
		return nil
	}
	for key := range statusBlock {
		for _, statDesc := range crd.StatusDescriptors {
			if statDesc.Path == key {
				test.earnedPoints++
				delete(statusBlock, key)
				break
			}
		}
	}
	scTests = append(scTests, test)
	for key := range statusBlock {
		scSuggestions = append(scSuggestions, "Add a status descriptor for "+key)
	}
	return nil
}

// specDescriptors makes sure that all spec fields found in the created CR has a matching descriptor in the CSV
func specDescriptors(csv *olmapiv1alpha1.ClusterServiceVersion, runtimeClient client.Client, obj *unstructured.Unstructured) error {
	test := scorecardTest{testType: olmIntegration, name: "Spec fields with descriptors"}
	err := runtimeClient.Get(context.TODO(), types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()}, obj)
	if err != nil {
		return err
	}
	if obj.Object["spec"] == nil {
		// what should we do if there is no spec block? Maybe some kind of N/A type output?
		scTests = append(scTests, test)
		return nil
	}
	specBlock := obj.Object["spec"].(map[string]interface{})
	test.maximumPoints = len(specBlock)
	var crd *olmapiv1alpha1.CRDDescription
	for _, owned := range csv.Spec.CustomResourceDefinitions.Owned {
		if owned.Kind == obj.GetKind() {
			crd = &owned
			break
		}
	}
	if crd == nil {
		scTests = append(scTests, test)
		return nil
	}
	for key := range specBlock {
		for _, specDesc := range crd.SpecDescriptors {
			if specDesc.Path == key {
				test.earnedPoints++
				delete(specBlock, key)
				break
			}
		}
	}
	scTests = append(scTests, test)
	for key := range specBlock {
		scSuggestions = append(scSuggestions, "Add a spec descriptor for "+key)
	}
	return nil
}
