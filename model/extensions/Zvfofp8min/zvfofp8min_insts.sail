//=======================================================================================
//  This Sail RISC-V architecture model, comprising all files and
//  directories except where otherwise noted is subject the BSD
//  two-clause license in the LICENSE file.
//
//  SPDX-License-Identifier: BSD-2-Clause
//=======================================================================================

// ******************************************************************************
// Zvfofp8min: OFP8 Conversion Extension, Version 0.9
//
// The OFP8 format is selected by vtype.altfmt when SEW=8:
//   altfmt=0 selects E4M3 and altfmt=1 selects E5M2.
//
// The SEW=8 forms of vfwcvtbf16.f.f.v and vfncvtbf16.f.f.w (whose SEW=16
// forms are part of Zvfbfmin) are implemented here and dispatched to from
// extensions/bfloat16/zvfbfmin_insts.sail.
// ******************************************************************************

function clause currentlyEnabled(Ext_Zvfofp8min) = hartSupports(Ext_Zvfofp8min) & currentlyEnabled(Ext_Zve32f)

private function get_ofp8_format() -> ofp8_format =
  if vtype[altfmt] == 0b1 then OFP8_E5M2 else OFP8_E4M3

// The floating-point format that is narrowed to OFP8.
enum ofp8_narrowing_source = {OFP8_FROM_BF16, OFP8_FROM_FP32}

// vfwcvtbf16.f.f.v with SEW=8: convert OFP8 to BF16.
function execute_vfwcvt_ofp8_to_bf16(vm : bits(1), vs2 : vregidx, vd : vregidx) -> ExecutionResult = {
  let rm_3b    = fcsr[FRM];
  let SEW      = get_sew();
  let LMUL_pow = get_lmul_pow();
  let num_elem = get_num_elem(LMUL_pow, SEW);

  let SEW_widen      = SEW * 2;
  let LMUL_pow_widen = LMUL_pow + 1;

  if illegal_vstart(VSTART_ARITH, num_elem, LMUL_pow) |
     illegal_fp_variable_width(vd, vm, SEW, rm_3b, SEW_widen, LMUL_pow_widen) |
     not(valid_reg_overlap(vs2, vd, LMUL_pow, LMUL_pow_widen)) |
     not(valid_vm_src_overlap(vm, vs2, SEW)) |
     SEW != 8
  then return Illegal_Instruction();

  assert(SEW == 8);
  assert(LMUL_pow_widen <= 3);

  let 'n = num_elem;
  let fmt = get_ofp8_format();

  let vm_val  : bits('n)             = read_vmask(num_elem, vm, zvreg);
  let vs2_val : vector('n, bits(8))  = read_vreg(num_elem, 8, LMUL_pow, vs2);
  let vd_val  : vector('n, bits(16)) = read_vreg(num_elem, 16, LMUL_pow_widen, vd);

  let (initial_result, mask) : (vector('n, bits(16)), bits('n)) = init_masked_result(num_elem, 16, LMUL_pow_widen, vd_val, vm_val);
  var result = initial_result;

  // "No rounding occurs, and no floating-point exception flags are set."
  foreach (i from 0 to (num_elem - 1)) {
    if mask[i] == 0b1 then result[i] = ofp8_to_bf16(fmt, vs2_val[i]);
  };

  write_vreg(num_elem, 16, LMUL_pow_widen, vd, result);
  set_vstart(zeros());
  RETIRE_SUCCESS
}

// Convert BF16 (EMUL=2*LMUL) or FP32 (EMUL=4*LMUL) to OFP8. This implements
// vfncvtbf16.f.f.w with SEW=8, vfncvtbf16.sat.f.f.w, vfncvt.f.f.q and
// vfncvt.sat.f.f.q.
function execute_vfncvt_to_ofp8(vm : bits(1), vs2 : vregidx, vd : vregidx, source : ofp8_narrowing_source, saturate : bool) -> ExecutionResult = {
  let rm_3b    = fcsr[FRM];
  let SEW      = get_sew();
  let LMUL_pow = get_lmul_pow();
  let num_elem = get_num_elem(LMUL_pow, SEW);

  let (SEW_source, LMUL_pow_source) : ({16, 32}, int) = match source {
    OFP8_FROM_BF16 => (16, LMUL_pow + 1),
    OFP8_FROM_FP32 => (32, LMUL_pow + 2),
  };

  if illegal_vstart(VSTART_ARITH, num_elem, LMUL_pow) |
     illegal_fp_variable_width(vd, vm, SEW, rm_3b, SEW_source, LMUL_pow_source) |
     not(valid_reg_overlap(vs2, vd, LMUL_pow_source, LMUL_pow)) |
     not(valid_vm_src_overlap(vm, vs2, SEW_source)) |
     SEW != 8
  then return Illegal_Instruction();

  assert(SEW == 8);
  assert(LMUL_pow_source <= 3);

  let 'n = num_elem;
  let fmt = get_ofp8_format();
  let rm  = encdec_rounding_mode(rm_3b);

  let vm_val : bits('n)            = read_vmask(num_elem, vm, zvreg);
  let vd_val : vector('n, bits(8)) = read_vreg(num_elem, 8, LMUL_pow, vd);

  let (initial_result, mask) : (vector('n, bits(8)), bits('n)) = init_masked_result(num_elem, 8, LMUL_pow, vd_val, vm_val);
  var result = initial_result;

  match source {
    OFP8_FROM_BF16 => {
      let vs2_val : vector('n, bits(16)) = read_vreg(num_elem, 16, LMUL_pow_source, vs2);
      foreach (i from 0 to (num_elem - 1)) {
        if mask[i] == 0b1 then {
          let (fflags, elem) = bf16_to_ofp8(fmt, saturate, rm, vs2_val[i]);
          accrue_fflags(fflags);
          result[i] = elem;
        }
      }
    },
    OFP8_FROM_FP32 => {
      let vs2_val : vector('n, bits(32)) = read_vreg(num_elem, 32, LMUL_pow_source, vs2);
      foreach (i from 0 to (num_elem - 1)) {
        if mask[i] == 0b1 then {
          let (fflags, elem) = f32_to_ofp8(fmt, saturate, rm, vs2_val[i]);
          accrue_fflags(fflags);
          result[i] = elem;
        }
      }
    },
  };

  write_vreg(num_elem, 8, LMUL_pow, vd, result);
  set_vstart(zeros());
  RETIRE_SUCCESS
}

// ************************* vfncvtbf16.sat.f.f.w *******************************
union clause instruction = VFNCVTBF16_SAT_F_F_W : (bits(1), vregidx, vregidx)

$[wavedrom "VFUNARY0 _ _ vfncvtbf16.sat OPFVV _ OP-V"]
mapping clause encdec = VFNCVTBF16_SAT_F_F_W(vm, vs2, vd)
  <-> 0b010010 @ vm @ encdec_vreg(vs2) @ 0b11111 @ 0b001 @ encdec_vreg(vd) @ 0b1010111
  when currentlyEnabled(Ext_Zvfofp8min) & get_sew() == 8

function clause execute VFNCVTBF16_SAT_F_F_W(vm, vs2, vd) =
  execute_vfncvt_to_ofp8(vm, vs2, vd, OFP8_FROM_BF16, true)

mapping clause assembly = VFNCVTBF16_SAT_F_F_W(vm, vs2, vd)
  <-> "vfncvtbf16.sat.f.f.w" ^ spc() ^ vreg_name(vd) ^ sep() ^ vreg_name(vs2) ^ sep() ^ maybe_vmask(vm)

// ***************************** vfncvt.f.f.q ***********************************
union clause instruction = VFNCVT_F_F_Q : (bits(1), vregidx, vregidx)

$[wavedrom "VFUNARY0 _ _ vfncvt.f.f.q OPFVV _ OP-V"]
mapping clause encdec = VFNCVT_F_F_Q(vm, vs2, vd)
  <-> 0b010010 @ vm @ encdec_vreg(vs2) @ 0b11001 @ 0b001 @ encdec_vreg(vd) @ 0b1010111
  when currentlyEnabled(Ext_Zvfofp8min) & get_sew() == 8

function clause execute VFNCVT_F_F_Q(vm, vs2, vd) =
  execute_vfncvt_to_ofp8(vm, vs2, vd, OFP8_FROM_FP32, false)

mapping clause assembly = VFNCVT_F_F_Q(vm, vs2, vd)
  <-> "vfncvt.f.f.q" ^ spc() ^ vreg_name(vd) ^ sep() ^ vreg_name(vs2) ^ sep() ^ maybe_vmask(vm)

// *************************** vfncvt.sat.f.f.q *********************************
union clause instruction = VFNCVT_SAT_F_F_Q : (bits(1), vregidx, vregidx)

$[wavedrom "VFUNARY0 _ _ vfncvt.sat.f.f.q OPFVV _ OP-V"]
mapping clause encdec = VFNCVT_SAT_F_F_Q(vm, vs2, vd)
  <-> 0b010010 @ vm @ encdec_vreg(vs2) @ 0b11011 @ 0b001 @ encdec_vreg(vd) @ 0b1010111
  when currentlyEnabled(Ext_Zvfofp8min) & get_sew() == 8

function clause execute VFNCVT_SAT_F_F_Q(vm, vs2, vd) =
  execute_vfncvt_to_ofp8(vm, vs2, vd, OFP8_FROM_FP32, true)

mapping clause assembly = VFNCVT_SAT_F_F_Q(vm, vs2, vd)
  <-> "vfncvt.sat.f.f.q" ^ spc() ^ vreg_name(vd) ^ sep() ^ vreg_name(vs2) ^ sep() ^ maybe_vmask(vm)
