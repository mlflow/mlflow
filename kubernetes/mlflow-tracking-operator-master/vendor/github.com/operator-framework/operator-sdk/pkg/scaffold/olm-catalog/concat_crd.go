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

package catalog

import (
	"io/ioutil"
	"path/filepath"

	"github.com/operator-framework/operator-sdk/internal/util/yamlutil"
	"github.com/operator-framework/operator-sdk/pkg/scaffold"
	"github.com/operator-framework/operator-sdk/pkg/scaffold/input"

	"github.com/spf13/afero"
)

const ConcatCRDYamlFile = "_generated.concat_crd.yaml"

// ConcatCRD scaffolds a file of all concatenated CRD's found using config file
// fields. This file is used by the OLM to create CR's in conjunction with the
// operators' CSV.
type ConcatCRD struct {
	input.Input

	// ConfigFilePath is the location of a configuration file path for this
	// projects' CSV file.
	ConfigFilePath string
}

func (s *ConcatCRD) GetInput() (input.Input, error) {
	if s.Path == "" {
		s.Path = filepath.Join(scaffold.OLMCatalogDir, ConcatCRDYamlFile)
	}
	if s.ConfigFilePath == "" {
		s.ConfigFilePath = filepath.Join(scaffold.OLMCatalogDir, CSVConfigYamlFile)
	}
	return s.Input, nil
}

func (s *ConcatCRD) SetFS(_ afero.Fs) {}

// CustomRender returns the bytes of all CRD manifests concatenated into one file.
func (s *ConcatCRD) CustomRender() ([]byte, error) {
	cfg, err := getCSVConfig(s.ConfigFilePath)
	if err != nil {
		return nil, err
	}
	return concatCRDsInPaths(cfg.CRDCRPaths)
}

// concatCRDsInPaths concatenates CRD manifests found at crdPaths into one
// file, delimited by `---`.
func concatCRDsInPaths(crdPaths []string) (cb []byte, err error) {
	for _, f := range crdPaths {
		yamlData, err := ioutil.ReadFile(f)
		if err != nil {
			return nil, err
		}

		scanner := yamlutil.NewYAMLScanner(yamlData)
		for scanner.Scan() {
			yamlSpec := scanner.Bytes()
			k, err := getKindfromYAML(yamlSpec)
			if err != nil {
				return nil, err
			}
			if k == "CustomResourceDefinition" {
				cb = yamlutil.CombineManifests(cb, yamlSpec)
			}
		}
	}

	return cb, nil
}
