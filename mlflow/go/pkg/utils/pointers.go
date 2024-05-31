package utils

import "strconv"

func PtrTo[T any](v T) *T {
	return &v
}

func ConvertInt32PointerToStringPointer(iPtr *int32) *string {
	if iPtr == nil {
		return nil
	}

	iValue := *iPtr
	sValue := strconv.Itoa(int(iValue))

	return &sValue
}
