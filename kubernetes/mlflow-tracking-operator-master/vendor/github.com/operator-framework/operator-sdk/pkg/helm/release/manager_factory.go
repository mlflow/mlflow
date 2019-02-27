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

package release

import (
	"fmt"
	"strings"

	"github.com/martinlindhe/base36"
	"github.com/pborman/uuid"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	apitypes "k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	helmengine "k8s.io/helm/pkg/engine"
	"k8s.io/helm/pkg/kube"
	"k8s.io/helm/pkg/storage"
	"k8s.io/helm/pkg/tiller"
	"k8s.io/helm/pkg/tiller/environment"

	"github.com/operator-framework/operator-sdk/pkg/helm/engine"
	"github.com/operator-framework/operator-sdk/pkg/helm/internal/types"
)

// ManagerFactory creates Managers that are specific to custom resources. It is
// used by the HelmOperatorReconciler during resource reconciliation, and it
// improves decoupling between reconciliation logic and the Helm backend
// components used to manage releases.
type ManagerFactory interface {
	NewManager(r *unstructured.Unstructured) Manager
}

type managerFactory struct {
	storageBackend   *storage.Storage
	tillerKubeClient *kube.Client
	chartDir         string
}

// NewManagerFactory returns a new Helm manager factory capable of installing and uninstalling releases.
func NewManagerFactory(storageBackend *storage.Storage, tillerKubeClient *kube.Client, chartDir string) ManagerFactory {
	return &managerFactory{storageBackend, tillerKubeClient, chartDir}
}

func (f managerFactory) NewManager(r *unstructured.Unstructured) Manager {
	return f.newManagerForCR(r)
}

func (f managerFactory) newManagerForCR(r *unstructured.Unstructured) Manager {
	return &manager{
		storageBackend:   f.storageBackend,
		tillerKubeClient: f.tillerKubeClient,
		chartDir:         f.chartDir,

		tiller:      f.tillerRendererForCR(r),
		releaseName: getReleaseName(r),
		namespace:   r.GetNamespace(),

		spec:   r.Object["spec"],
		status: types.StatusFor(r),
	}
}

// tillerRendererForCR creates a ReleaseServer configured with a rendering engine that adds ownerrefs to rendered assets
// based on the CR.
func (f managerFactory) tillerRendererForCR(r *unstructured.Unstructured) *tiller.ReleaseServer {
	controllerRef := metav1.NewControllerRef(r, r.GroupVersionKind())
	ownerRefs := []metav1.OwnerReference{
		*controllerRef,
	}
	baseEngine := helmengine.New()
	e := engine.NewOwnerRefEngine(baseEngine, ownerRefs)
	var ey environment.EngineYard = map[string]environment.Engine{
		environment.GoTplEngine: e,
	}
	env := &environment.Environment{
		EngineYard: ey,
		Releases:   f.storageBackend,
		KubeClient: f.tillerKubeClient,
	}
	kubeconfig, _ := f.tillerKubeClient.ToRESTConfig()
	cs := clientset.NewForConfigOrDie(kubeconfig)

	return tiller.NewReleaseServer(env, cs, false)
}

func getReleaseName(r *unstructured.Unstructured) string {
	return fmt.Sprintf("%s-%s", r.GetName(), shortenUID(r.GetUID()))
}

func shortenUID(uid apitypes.UID) string {
	u := uuid.Parse(string(uid))
	uidBytes, err := u.MarshalBinary()
	if err != nil {
		return strings.Replace(string(uid), "-", "", -1)
	}
	return strings.ToLower(base36.EncodeBytes(uidBytes))
}
