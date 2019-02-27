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

package generate

import (
	"fmt"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"strings"

	genutil "github.com/operator-framework/operator-sdk/commands/operator-sdk/cmd/generate/internal"
	"github.com/operator-framework/operator-sdk/internal/util/projutil"
	"github.com/operator-framework/operator-sdk/pkg/scaffold"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

func NewGenerateK8SCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "k8s",
		Short: "Generates Kubernetes code for custom resource",
		Long: `k8s generator generates code for custom resources given the API
specs in pkg/apis/<group>/<version> directories to comply with kube-API
requirements. Go code is generated under
pkg/apis/<group>/<version>/zz_generated.deepcopy.go.

Example:
	$ operator-sdk generate k8s
	$ tree pkg/apis
	pkg/apis/
	└── app
		└── v1alpha1
			├── zz_generated.deepcopy.go
`,
		RunE: k8sFunc,
	}
}

func k8sFunc(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return fmt.Errorf("command %s doesn't accept any arguments", cmd.CommandPath())
	}

	// Only Go projects can generate k8s deepcopy code.
	if err := projutil.CheckGoProjectCmd(cmd); err != nil {
		return err
	}

	return K8sCodegen()
}

// K8sCodegen performs deepcopy code-generation for all custom resources under pkg/apis
func K8sCodegen() error {
	projutil.MustInProjectRoot()

	wd := projutil.MustGetwd()
	repoPkg := projutil.CheckAndGetProjectGoPkg()
	srcDir := filepath.Join(wd, "vendor", "k8s.io", "code-generator")
	binDir := filepath.Join(wd, scaffold.BuildBinDir)

	if err := buildCodegenBinaries(binDir, srcDir); err != nil {
		return err
	}

	gvMap, err := genutil.ParseGroupVersions()
	if err != nil {
		return fmt.Errorf("failed to parse group versions: (%v)", err)
	}
	gvb := &strings.Builder{}
	for g, vs := range gvMap {
		gvb.WriteString(fmt.Sprintf("%s:%v, ", g, vs))
	}

	log.Infof("Running deepcopy code-generation for Custom Resource group versions: [%v]\n", gvb.String())

	if err := deepcopyGen(binDir, repoPkg, gvMap); err != nil {
		return err
	}

	if err = defaulterGen(binDir, repoPkg, gvMap); err != nil {
		return err
	}

	log.Info("Code-generation complete.")
	return nil
}

func buildCodegenBinaries(binDir, codegenSrcDir string) error {
	genDirs := []string{
		"./cmd/defaulter-gen",
		"./cmd/client-gen",
		"./cmd/lister-gen",
		"./cmd/informer-gen",
		"./cmd/deepcopy-gen",
	}
	return genutil.BuildCodegenBinaries(genDirs, binDir, codegenSrcDir)
}

func deepcopyGen(binDir, repoPkg string, gvMap map[string][]string) (err error) {
	apisPkg := filepath.Join(repoPkg, scaffold.ApisDir)
	args := []string{
		"--input-dirs", genutil.CreateFQApis(apisPkg, gvMap),
		"--output-file-base", "zz_generated.deepcopy",
		"--bounding-dirs", apisPkg,
	}
	cmd := exec.Command(filepath.Join(binDir, "deepcopy-gen"), args...)
	if projutil.IsGoVerbose() {
		err = projutil.ExecCmd(cmd)
	} else {
		cmd.Stdout = ioutil.Discard
		cmd.Stderr = ioutil.Discard
		err = cmd.Run()
	}
	if err != nil {
		return fmt.Errorf("failed to perform deepcopy code-generation: %v", err)
	}
	return nil
}

func defaulterGen(binDir, repoPkg string, gvMap map[string][]string) (err error) {
	apisPkg := filepath.Join(repoPkg, scaffold.ApisDir)
	args := []string{
		"--input-dirs", genutil.CreateFQApis(apisPkg, gvMap),
		"--output-file-base", "zz_generated.defaults",
	}
	cmd := exec.Command(filepath.Join(binDir, "defaulter-gen"), args...)
	if projutil.IsGoVerbose() {
		err = projutil.ExecCmd(cmd)
	} else {
		cmd.Stdout = ioutil.Discard
		cmd.Stderr = ioutil.Discard
		err = cmd.Run()
	}
	if err != nil {
		return fmt.Errorf("failed to perform defaulter code-generation: %v", err)
	}
	return nil
}
