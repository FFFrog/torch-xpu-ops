skip_dict = {
    # SYCL default context is not supported on Windows
    "test_tensor_creation_ops_xpu.py": (
        "test_alias_from_dlpack_xpu_bfloat16",
        "test_alias_from_dlpack_xpu_complex128",
        "test_alias_from_dlpack_xpu_complex64",
        "test_alias_from_dlpack_xpu_float16",
        "test_alias_from_dlpack_xpu_float32",
        "test_alias_from_dlpack_xpu_float64",
        "test_alias_from_dlpack_xpu_int16",
        "test_alias_from_dlpack_xpu_int32",
        "test_alias_from_dlpack_xpu_int64",
        "test_alias_from_dlpack_xpu_int8",
        "test_alias_from_dlpack_xpu_uint8",
        "test_copy_from_dlpack_xpu_bfloat16",
        "test_copy_from_dlpack_xpu_complex128",
        "test_copy_from_dlpack_xpu_complex64",
        "test_copy_from_dlpack_xpu_float16",
        "test_copy_from_dlpack_xpu_float32",
        "test_copy_from_dlpack_xpu_float64",
        "test_copy_from_dlpack_xpu_int16",
        "test_copy_from_dlpack_xpu_int32",
        "test_copy_from_dlpack_xpu_int64",
        "test_copy_from_dlpack_xpu_int8",
        "test_copy_from_dlpack_xpu_uint8"
    ),
    # SYCL compiler issue where host and device results differ for math ops with complex dtypes
    "test_unary_ufuncs_xpu.py": (
        "test_reference_numerics_extremal__refs_atanh_xpu_complex128",
        "test_reference_numerics_extremal__refs_atanh_xpu_complex64",
        "test_reference_numerics_extremal__refs_nn_functional_tanhshrink_xpu_complex128",
        "test_reference_numerics_extremal__refs_sin_xpu_complex128",
        "test_reference_numerics_extremal__refs_sin_xpu_complex64",
        "test_reference_numerics_extremal__refs_sinh_xpu_complex128",
        "test_reference_numerics_extremal__refs_sinh_xpu_complex64",
        "test_reference_numerics_extremal__refs_tan_xpu_complex128",
        "test_reference_numerics_extremal__refs_tan_xpu_complex64",
        "test_reference_numerics_extremal_atanh_xpu_complex128",
        "test_reference_numerics_extremal_atanh_xpu_complex64",
        "test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex128",
        "test_reference_numerics_extremal_sin_xpu_complex128",
        "test_reference_numerics_extremal_sin_xpu_complex64",
        "test_reference_numerics_extremal_sinh_xpu_complex128",
        "test_reference_numerics_extremal_sinh_xpu_complex64",
        "test_reference_numerics_extremal_square_xpu_complex128",
        "test_reference_numerics_extremal_square_xpu_complex64",
        "test_reference_numerics_extremal_tan_xpu_complex128",
        "test_reference_numerics_extremal_tan_xpu_complex64",
        "test_reference_numerics_large__refs_cos_xpu_complex128",
        "test_reference_numerics_large__refs_cos_xpu_complex32",
        "test_reference_numerics_large__refs_cos_xpu_complex64",
        "test_reference_numerics_large__refs_cosh_xpu_complex32",
        "test_reference_numerics_large__refs_exp_xpu_complex128",
        "test_reference_numerics_large__refs_exp_xpu_complex32",
        "test_reference_numerics_large__refs_exp_xpu_complex64",
        "test_reference_numerics_large__refs_sin_xpu_complex128",
        "test_reference_numerics_large__refs_sin_xpu_complex32",
        "test_reference_numerics_large__refs_sin_xpu_complex64",
        "test_reference_numerics_large__refs_sinh_xpu_complex32",
        "test_reference_numerics_large__refs_tan_xpu_complex32",
        "test_reference_numerics_large_cos_xpu_complex128",
        "test_reference_numerics_large_cos_xpu_complex32",
        "test_reference_numerics_large_cos_xpu_complex64",
        "test_reference_numerics_large_exp_xpu_complex128",
        "test_reference_numerics_large_exp_xpu_complex64",
        "test_reference_numerics_large_sin_xpu_complex128",
        "test_reference_numerics_large_sin_xpu_complex32",
        "test_reference_numerics_large_sin_xpu_complex64",
        "test_reference_numerics_small_acos_xpu_complex32",
    ),
    "test_ops_xpu.py": (
        "test_compare_cpu_pow_xpu_bfloat16",  # https://github.com/intel/torch-xpu-ops/pull/764
    ),

}