package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// TrackingServerSpec defines the desired state of TrackingServer
// +k8s:openapi-gen=true
type TrackingServerSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "operator-sdk generate k8s" to regenerate code after modifying this file
	// Add custom validation using kubebuilder tags: https://book.kubebuilder.io/beyond_basics/generating_crd.html
	Size int32 `json:"size"`
	Image string `json"image"`
	S3_ENDPOINT_URL string `json"s3_endpoint_url"`
	AWS_SECRET_NAME string `json"aws_cred_secret"`
}

// TrackingServerStatus defines the observed state of TrackingServer
// +k8s:openapi-gen=true
type TrackingServerStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "operator-sdk generate k8s" to regenerate code after modifying this file
	// Add custom validation using kubebuilder tags: https://book.kubebuilder.io/beyond_basics/generating_crd.html
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TrackingServer is the Schema for the trackingservers API
// +k8s:openapi-gen=true
type TrackingServer struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TrackingServerSpec   `json:"spec,omitempty"`
	Status TrackingServerStatus `json:"status,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TrackingServerList contains a list of TrackingServer
type TrackingServerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TrackingServer `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TrackingServer{}, &TrackingServerList{})
}
