// RUN: spyre-triton-opt %s -split-input-file --lower-descriptor-memory -verify-diagnostics

// A gather/scatter's index buffer gets a memory view with its descriptor's
// declared signedness (si32), which dataflow-scheduler's IAB requires. That
// view belongs to the whole descriptor, so the pass rejects an index buffer
// that is also accessed as data: the data access would see memref<...xsi32>
// against a signless tensor, the mismatch dbo-opt's ktdp.store lowering
// rejects and the signless rule for every other view exists to prevent.

// -----
// Stored to, as well as gathered with.

tt.func @index_buffer_also_stored(%ptr: !tt.ptr<f16>, %idx_ptr: !tt.ptr<i32>,
                                  %y_offset: i32, %vals: tensor<32xi32>)
    -> tensor<32x64xf16> {
  %M = arith.constant 1024 : i32
  %K = arith.constant 128 : i32
  %sr = arith.constant 128 : i64
  %sc = arith.constant 1 : i64
  %ic = arith.constant 32 : i32
  %is = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  // expected-note @below {{index buffer descriptor here}}
  %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%ic], [%is] : <i32>, <32xsi32>
  %x_offsets = tt.descriptor_load %idx_desc[%c0] : !tt.tensordesc<32xsi32> -> tensor<32xi32>
  %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%sr, %sc] : <f16>, <1x64xf16>
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
  // expected-error @below {{cannot also be used as data}}
  tt.descriptor_store %idx_desc[%c0], %vals : !tt.tensordesc<32xsi32>, tensor<32xi32>
  tt.return %data : tensor<32x64xf16>
}

// -----
// Loaded once, and the loaded indices used as data besides x_offsets.

tt.func @index_values_also_returned(%ptr: !tt.ptr<f16>, %idx_ptr: !tt.ptr<i32>,
                                    %y_offset: i32)
    -> (tensor<32x64xf16>, tensor<32xi32>) {
  %M = arith.constant 1024 : i32
  %K = arith.constant 128 : i32
  %sr = arith.constant 128 : i64
  %sc = arith.constant 1 : i64
  %ic = arith.constant 32 : i32
  %is = arith.constant 1 : i64
  %c0 = arith.constant 0 : i32
  // expected-note @below {{index buffer descriptor here}}
  %idx_desc = tt.make_tensor_descriptor %idx_ptr, [%ic], [%is] : <i32>, <32xsi32>
  %x_offsets = tt.descriptor_load %idx_desc[%c0] : !tt.tensordesc<32xsi32> -> tensor<32xi32>
  %desc = tt.make_tensor_descriptor %ptr, [%M, %K], [%sr, %sc] : <f16>, <1x64xf16>
  %data = tt.descriptor_gather %desc[%x_offsets, %y_offset]
      : (!tt.tensordesc<1x64xf16>, tensor<32xi32>, i32) -> tensor<32x64xf16>
  // expected-error @below {{cannot also be used as data}}
  tt.return %data, %x_offsets : tensor<32x64xf16>, tensor<32xi32>
}
