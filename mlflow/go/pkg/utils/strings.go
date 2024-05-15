package utils

func IsNotNilOrEmptyString(v *string) bool {
	return v != nil && *v != ""
}

func IsNilOrEmptyString(v *string) bool {
	return v == nil || *v == ""
}
