package trackingserver

import (
	"context"
	"k8s.io/api/apps/v1"

	aiv1alpha1 "github.com/zmhassan/mlflow-tracking-operator/pkg/apis/ai/v1alpha1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
	"sigs.k8s.io/controller-runtime/pkg/source"
)

var log = logf.Log.WithName("controller_trackingserver")

/**
* USER ACTION REQUIRED: This is a scaffold file intended for the user to modify with their own Controller
* business logic.  Delete these comments after modifying this file.*
 */

// Add creates a new TrackingServer Controller and adds it to the Manager. The Manager will set fields on the Controller
// and Start it when the Manager is Started.
func Add(mgr manager.Manager) error {
	return add(mgr, newReconciler(mgr))
}

// newReconciler returns a new reconcile.Reconciler
func newReconciler(mgr manager.Manager) reconcile.Reconciler {
	return &ReconcileTrackingServer{client: mgr.GetClient(), scheme: mgr.GetScheme()}
}

// add adds a new Controller to mgr with r as the reconcile.Reconciler
func add(mgr manager.Manager, r reconcile.Reconciler) error {
	// Create a new controller
	c, err := controller.New("trackingserver-controller", mgr, controller.Options{Reconciler: r})
	if err != nil {
		return err
	}

	// Watch for changes to primary resource TrackingServer
	err = c.Watch(&source.Kind{Type: &aiv1alpha1.TrackingServer{}}, &handler.EnqueueRequestForObject{})
	if err != nil {
		return err
	}

	// TODO(user): Modify this to be the types you create that are owned by the primary resource
	// Watch for changes to secondary resource Pods and requeue the owner TrackingServer
	err = c.Watch(&source.Kind{Type: &corev1.Pod{}}, &handler.EnqueueRequestForOwner{
		IsController: true,
		OwnerType:    &aiv1alpha1.TrackingServer{},
	})
	if err != nil {
		return err
	}

	err = c.Watch(&source.Kind{Type: &corev1.Service{}}, &handler.EnqueueRequestForOwner{
		IsController: true,
		OwnerType:    &aiv1alpha1.TrackingServer{},
	})
	if err != nil {
		return err
	}

	return nil
}

var _ reconcile.Reconciler = &ReconcileTrackingServer{}

// ReconcileTrackingServer reconciles a TrackingServer object
type ReconcileTrackingServer struct {
	// This client, initialized using mgr.Client() above, is a split client
	// that reads objects from the cache and writes to the apiserver
	client client.Client
	scheme *runtime.Scheme
}

// Reconcile reads that state of the cluster for a TrackingServer object and makes changes based on the state read
// and what is in the TrackingServer.Spec
// TODO(user): Modify this Reconcile function to implement your Controller logic.  This example creates
// a Pod as an example
// Note:
// The Controller will requeue the Request to be processed again if the returned error is non-nil or
// Result.Requeue is true, otherwise upon completion it will remove the work from the queue.
func (r *ReconcileTrackingServer) Reconcile(request reconcile.Request) (reconcile.Result, error) {
	reqLogger := log.WithValues("Request.Namespace", request.Namespace, "Request.Name", request.Name)
	reqLogger.Info("Reconciling TrackingServer")

	// Fetch the TrackingServer instance
	instance := &aiv1alpha1.TrackingServer{}
	err := r.client.Get(context.TODO(), request.NamespacedName, instance)
	if err != nil {
		if errors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			return reconcile.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return reconcile.Result{}, err
	}

	// Define a new Deployment object
	var deployment = newDeploymentForMLFlow(instance)
	srv := newServiceForMLFlow(instance)

	// Set TrackingServer instance as the owner and controller
	if err := controllerutil.SetControllerReference(instance, deployment, r.scheme); err != nil {
		return reconcile.Result{}, err
	}

	if err := controllerutil.SetControllerReference(instance, srv, r.scheme); err != nil {
		return reconcile.Result{}, err
	}

	// Check if this Deployment already exists
	found := &v1.Deployment{}
	err = r.client.Get(context.TODO(), types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		reqLogger.Info("Creating a new Deployment", "Deployment.Namespace", deployment.Namespace, "Deployment.Name", deployment.Name)

		err = r.client.Create(context.TODO(), deployment)
		if err != nil {
			return reconcile.Result{}, err
		}
		// Creating a service to map to service
		err2 := r.client.Create(context.TODO(), srv)
		if err2 != nil {
			return reconcile.Result{}, err2
		}
		// Deployment created successfully - don't requeue
		return reconcile.Result{}, nil
	} else if err != nil {
		return reconcile.Result{}, err
	}

	// Deployment already exists - don't requeue
	reqLogger.Info("Skip reconcile: Deployment already exists", "Deployment.Namespace", found.Namespace, "Deployment.Name", found.Name)
	return reconcile.Result{}, nil
}

func newServiceForMLFlow(cr *aiv1alpha1.TrackingServer) *corev1.Service  {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: cr.Name+"-svc",
			Namespace: cr.Namespace,
			Labels: map[string]string{
				"app": cr.Name+"-svc",
				"type": "service",
			},
			OwnerReferences: []metav1.OwnerReference{},
		},
		Spec: corev1.ServiceSpec{
			Type:      "ClusterIP",
			ClusterIP: "None",
			Selector: map[string]string{
				"app": cr.Name,
			},
			Ports: []corev1.ServicePort{{
				Name: "http",
				Port: 5000,
			}},
		},
	}
	return service
}

func  newDeploymentForMLFlow(cr *aiv1alpha1.TrackingServer) *v1.Deployment{
	replicas := cr.Spec.Size
	labels := map[string]string{
		"app": cr.Name,
	}
	con:=[]corev1.Container{{
		Image:   cr.Spec.Image,
		Name:    cr.Name,
		Ports: []corev1.ContainerPort{{
			ContainerPort: 5000,
			Name:          "trackingserver",
		}},
	}}

	if len(cr.Spec.AWS_SECRET_NAME) != 0{
		con[0].EnvFrom= [] corev1.EnvFromSource{{
			SecretRef:&corev1.SecretEnvSource{ LocalObjectReference: corev1.LocalObjectReference{ Name: cr.Spec.AWS_SECRET_NAME}},
		}}
	}

	if len(cr.Spec.S3_ENDPOINT_URL) != 0{
		con[0].Env=[]corev1.EnvVar{{
			Name: "MLFLOW_S3_ENDPOINT_URL",
			Value: cr.Spec.S3_ENDPOINT_URL,
		}}
	}


	dep:= &v1.Deployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "apps/v1",
			Kind:       "Deployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      cr.Name,
			Namespace: cr.Namespace,
		},
		Spec: v1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					Containers: con,
				},
			},
		},
	}

	return dep

}

