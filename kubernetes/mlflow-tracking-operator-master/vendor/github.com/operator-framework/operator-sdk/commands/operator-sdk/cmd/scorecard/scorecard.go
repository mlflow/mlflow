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
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/operator-framework/operator-sdk/internal/util/projutil"
	"github.com/operator-framework/operator-sdk/pkg/scaffold"

	k8sInternal "github.com/operator-framework/operator-sdk/internal/util/k8sutil"
	"github.com/operator-framework/operator-sdk/internal/util/yamlutil"

	olmapiv1alpha1 "github.com/operator-framework/operator-lifecycle-manager/pkg/api/apis/operators/v1alpha1"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	v1 "k8s.io/api/core/v1"
	extscheme "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/kubernetes"
	cgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	ConfigOpt             = "config"
	NamespaceOpt          = "namespace"
	KubeconfigOpt         = "kubeconfig"
	InitTimeoutOpt        = "init-timeout"
	CSVPathOpt            = "csv-path"
	BasicTestsOpt         = "basic-tests"
	OLMTestsOpt           = "olm-tests"
	TenantTestsOpt        = "good-tenant-tests"
	NamespacedManifestOpt = "namespace-manifest"
	GlobalManifestOpt     = "global-manifest"
	CRManifestOpt         = "cr-manifest"
	ProxyImageOpt         = "proxy-image"
	ProxyPullPolicyOpt    = "proxy-pull-policy"
	CRDsDirOpt            = "crds-dir"
	VerboseOpt            = "verbose"
)

const (
	basicOperator  = "Basic Operator"
	olmIntegration = "OLM Integration"
	goodTenant     = "Good Tenant"
)

// TODO: add point weights to tests
type scorecardTest struct {
	testType      string
	name          string
	description   string
	earnedPoints  int
	maximumPoints int
}

type cleanupFn func() error

var (
	kubeconfig     *rest.Config
	scTests        []scorecardTest
	scSuggestions  []string
	dynamicDecoder runtime.Decoder
	runtimeClient  client.Client
	restMapper     *restmapper.DeferredDiscoveryRESTMapper
	deploymentName string
	proxyPod       *v1.Pod
	cleanupFns     []cleanupFn
	ScorecardConf  string
)

const scorecardPodName = "operator-scorecard-test"

func ScorecardTests(cmd *cobra.Command, args []string) error {
	err := initConfig()
	if err != nil {
		return err
	}
	if viper.GetString(CRManifestOpt) == "" {
		return errors.New("cr-manifest config option missing")
	}
	if !viper.GetBool(BasicTestsOpt) && !viper.GetBool(OLMTestsOpt) {
		return errors.New("at least one test type is required")
	}
	if viper.GetBool(OLMTestsOpt) && viper.GetString(CSVPathOpt) == "" {
		return fmt.Errorf("if olm-tests is enabled, the --csv-path flag must be set")
	}
	pullPolicy := viper.GetString(ProxyPullPolicyOpt)
	if pullPolicy != "Always" && pullPolicy != "Never" && pullPolicy != "PullIfNotPresent" {
		return fmt.Errorf("invalid proxy pull policy: (%s); valid values: Always, Never, PullIfNotPresent", pullPolicy)
	}
	cmd.SilenceUsage = true
	if viper.GetBool(VerboseOpt) {
		log.SetLevel(log.DebugLevel)
	}
	// if no namespaced manifest path is given, combine deploy/service_account.yaml, deploy/role.yaml, deploy/role_binding.yaml and deploy/operator.yaml
	if viper.GetString(NamespacedManifestOpt) == "" {
		file, err := yamlutil.GenerateCombinedNamespacedManifest(scaffold.DeployDir)
		if err != nil {
			return err
		}
		viper.Set(NamespacedManifestOpt, file.Name())
		defer func() {
			err := os.Remove(viper.GetString(NamespacedManifestOpt))
			if err != nil {
				log.Errorf("Could not delete temporary namespace manifest file: (%v)", err)
			}
		}()
	}
	if viper.GetString(GlobalManifestOpt) == "" {
		file, err := yamlutil.GenerateCombinedGlobalManifest(scaffold.CRDsDir)
		if err != nil {
			return err
		}
		viper.Set(GlobalManifestOpt, file.Name())
		defer func() {
			err := os.Remove(viper.GetString(GlobalManifestOpt))
			if err != nil {
				log.Errorf("Could not delete global manifest file: (%v)", err)
			}
		}()
	}
	defer func() {
		if err := cleanupScorecard(); err != nil {
			log.Errorf("Failed to clenup resources: (%v)", err)
		}
	}()
	var tmpNamespaceVar string
	kubeconfig, tmpNamespaceVar, err = k8sInternal.GetKubeconfigAndNamespace(viper.GetString(KubeconfigOpt))
	if err != nil {
		return fmt.Errorf("failed to build the kubeconfig: %v", err)
	}
	if viper.GetString(NamespaceOpt) == "" {
		viper.Set(NamespaceOpt, tmpNamespaceVar)
	}
	scheme := runtime.NewScheme()
	// scheme for client go
	if err := cgoscheme.AddToScheme(scheme); err != nil {
		return fmt.Errorf("failed to add client-go scheme to client: (%v)", err)
	}
	// api extensions scheme (CRDs)
	if err := extscheme.AddToScheme(scheme); err != nil {
		return fmt.Errorf("failed to add failed to add extensions api scheme to client: (%v)", err)
	}
	// olm api (CS
	if err := olmapiv1alpha1.AddToScheme(scheme); err != nil {
		return fmt.Errorf("failed to add failed to add oml api scheme (CSVs) to client: (%v)", err)
	}
	dynamicDecoder = serializer.NewCodecFactory(scheme).UniversalDeserializer()
	// if a user creates a new CRD, we need to be able to reset the rest mapper
	// temporary kubeclient to get a cached discovery
	kubeclient, err := kubernetes.NewForConfig(kubeconfig)
	if err != nil {
		return fmt.Errorf("failed to get a kubeclient: %v", err)
	}
	cachedDiscoveryClient := cached.NewMemCacheClient(kubeclient.Discovery())
	restMapper = restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscoveryClient)
	restMapper.Reset()
	runtimeClient, _ = client.New(kubeconfig, client.Options{Scheme: scheme, Mapper: restMapper})
	if err := createFromYAMLFile(viper.GetString(GlobalManifestOpt)); err != nil {
		return fmt.Errorf("failed to create global resources: %v", err)
	}
	if err := createFromYAMLFile(viper.GetString(NamespacedManifestOpt)); err != nil {
		return fmt.Errorf("failed to create namespaced resources: %v", err)
	}
	if err := createFromYAMLFile(viper.GetString(CRManifestOpt)); err != nil {
		return fmt.Errorf("failed to create cr resource: %v", err)
	}
	obj, err := yamlToUnstructured(viper.GetString(CRManifestOpt))
	if err != nil {
		return fmt.Errorf("failed to decode custom resource manifest into object: %s", err)
	}
	if viper.GetBool(BasicTestsOpt) {
		fmt.Println("Checking for existence of spec and status blocks in CR")
		err = checkSpecAndStat(runtimeClient, obj, false)
		if err != nil {
			return err
		}
		// This test is far too inconsistent and unreliable to be meaningful,
		// so it has been disabled
		/*
			fmt.Println("Checking that operator actions are reflected in status")
			err = checkStatusUpdate(runtimeClient, obj)
			if err != nil {
				return err
			}
		*/
		fmt.Println("Checking that writing into CRs has an effect")
		logs, err := writingIntoCRsHasEffect(obj)
		if err != nil {
			return err
		}
		log.Debugf("Scorecard Proxy Logs: %v\n", logs)
	} else {
		// checkSpecAndStat is used to make sure the operator is ready in this case
		// the boolean argument set at the end tells the function not to add the result to scTests
		err = checkSpecAndStat(runtimeClient, obj, true)
		if err != nil {
			return err
		}
	}
	if viper.GetBool(OLMTestsOpt) {
		yamlSpec, err := ioutil.ReadFile(viper.GetString(CSVPathOpt))
		if err != nil {
			return fmt.Errorf("failed to read csv: %v", err)
		}
		rawCSV, _, err := dynamicDecoder.Decode(yamlSpec, nil, nil)
		if err != nil {
			return err
		}
		csv := &olmapiv1alpha1.ClusterServiceVersion{}
		switch o := rawCSV.(type) {
		case *olmapiv1alpha1.ClusterServiceVersion:
			csv = o
		default:
			return fmt.Errorf("provided yaml file not of ClusterServiceVersion type")
		}
		fmt.Println("Checking if all CRDs have validation")
		if err := crdsHaveValidation(viper.GetString(CRDsDirOpt), runtimeClient, obj); err != nil {
			return err
		}
		fmt.Println("Checking for CRD resources")
		crdsHaveResources(csv)
		fmt.Println("Checking for existence of example CRs")
		annotationsContainExamples(csv)
		fmt.Println("Checking spec descriptors")
		err = specDescriptors(csv, runtimeClient, obj)
		if err != nil {
			return err
		}
		fmt.Println("Checking status descriptors")
		err = statusDescriptors(csv, runtimeClient, obj)
		if err != nil {
			return err
		}
	}
	var totalEarned, totalMax int
	var enabledTestTypes []string
	if viper.GetBool(BasicTestsOpt) {
		enabledTestTypes = append(enabledTestTypes, basicOperator)
	}
	if viper.GetBool(OLMTestsOpt) {
		enabledTestTypes = append(enabledTestTypes, olmIntegration)
	}
	if viper.GetBool(TenantTestsOpt) {
		enabledTestTypes = append(enabledTestTypes, goodTenant)
	}
	for _, testType := range enabledTestTypes {
		fmt.Printf("%s:\n", testType)
		for _, test := range scTests {
			if test.testType == testType {
				if !(test.earnedPoints == 0 && test.maximumPoints == 0) {
					fmt.Printf("\t%s: %d/%d points\n", test.name, test.earnedPoints, test.maximumPoints)
				} else {
					fmt.Printf("\t%s: N/A (depends on an earlier test that failed)\n", test.name)
				}
				totalEarned += test.earnedPoints
				totalMax += test.maximumPoints
			}
		}
	}
	fmt.Printf("\nTotal Score: %d/%d points\n", totalEarned, totalMax)
	for _, suggestion := range scSuggestions {
		// 33 is yellow (specifically, the same shade of yellow that logrus uses for warnings)
		fmt.Printf("\x1b[%dmSUGGESTION:\x1b[0m %s\n", 33, suggestion)
	}
	return nil
}

func initConfig() error {
	if ScorecardConf != "" {
		// Use config file from the flag.
		viper.SetConfigFile(ScorecardConf)
	} else {
		viper.AddConfigPath(projutil.MustGetwd())
		// using SetConfigName allows users to use a .yaml, .json, or .toml file
		viper.SetConfigName(".osdk-scorecard")
	}

	if err := viper.ReadInConfig(); err == nil {
		log.Info("Using config file: ", viper.ConfigFileUsed())
	} else {
		log.Warn("Could not load config file; using flags")
	}
	return nil
}
