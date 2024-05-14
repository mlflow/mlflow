package utils

func PtrTo[T any](v T) *T {
	return &v
}

func IsNotNilOrEmptyString(v *string) bool {
	return v != nil && *v != ""
}

func IsNilOrEmptyString(v *string) bool {
	return v == nil || *v == ""
}
