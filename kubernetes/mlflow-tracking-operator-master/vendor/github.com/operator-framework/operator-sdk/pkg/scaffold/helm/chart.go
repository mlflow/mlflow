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

package helm

import (
	"os"
	"path/filepath"

	"github.com/operator-framework/operator-sdk/pkg/scaffold"

	log "github.com/sirupsen/logrus"
	"k8s.io/helm/pkg/chartutil"
	"k8s.io/helm/pkg/proto/hapi/chart"
)

// HelmChartsDir is the relative directory within an SDK project where Helm
// charts are stored.
const HelmChartsDir string = "helm-charts"

// CreateChartForResource creates a new helm chart in the SDK project for the
// provided resource.
func CreateChartForResource(r *scaffold.Resource, projectDir string) (*chart.Chart, error) {
	log.Infof("Created %s/%s/", HelmChartsDir, r.LowerKind)

	chartfile := &chart.Metadata{
		Name:        r.LowerKind,
		Description: "A Helm chart for Kubernetes",
		Version:     "0.1.0",
		AppVersion:  "1.0",
		ApiVersion:  chartutil.ApiVersionV1,
	}

	chartsDir := filepath.Join(projectDir, HelmChartsDir)
	if err := os.MkdirAll(chartsDir, 0755); err != nil {
		return nil, err
	}
	chartDir, err := chartutil.Create(chartfile, chartsDir)
	if err != nil {
		return nil, err
	}
	return chartutil.LoadDir(chartDir)
}
