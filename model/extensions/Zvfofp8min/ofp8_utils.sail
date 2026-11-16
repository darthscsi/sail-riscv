//=======================================================================================
//  This Sail RISC-V architecture model, comprising all files and
//  directories except where otherwise noted is subject the BSD
//  two-clause license in the LICENSE file.
//
//  SPDX-License-Identifier: BSD-2-Clause
//=======================================================================================

// ******************************************************************************
// OCP 8-bit floating-point (OFP8) formats, as used by Zvfofp8min.
//
//          | sign | exponent | fraction | bias | NaN            | Infinity
//   E4M3   |  1   |    4     |    3     |  7   | S.1111.111     | none
//   E5M2   |  1   |    5     |    2     |  15  | S.11111.{01,10,11} | S.11111.00
//
// The canonical NaN of both formats is 0x7f.
// ******************************************************************************

enum ofp8_format = {OFP8_E4M3, OFP8_E5M2}

function ofp8_canonical_NaN() -> bits(8) = 0x7F

// Number of fraction bits.
private function ofp8_frac_bits(fmt : ofp8_format) -> {2, 3} =
  match fmt {
    OFP8_E4M3 => 3,
    OFP8_E5M2 => 2,
  }

// Exponent bias.
private function ofp8_bias(fmt : ofp8_format) -> {7, 15} =
  match fmt {
    OFP8_E4M3 => 7,
    OFP8_E5M2 => 15,
  }

// Encoding of the maximum-magnitude finite value, without the sign bit.
// E4M3: 1.75 * 2^8 (S.1111.110), E5M2: 1.75 * 2^15 (S.11110.11).
private function ofp8_max_finite(fmt : ofp8_format) -> bits(7) =
  match fmt {
    OFP8_E4M3 => 0b1111110,
    OFP8_E5M2 => 0b1111011,
  }

// Convert an OFP8 value to BF16. Every OFP8 value is exactly representable
// in BF16, so no rounding occurs and no exception flags are raised (not
// even for E5M2 signaling NaNs).
function ofp8_to_bf16(fmt : ofp8_format, v : bits(8)) -> bits(16) = {
  let sign = v[7 .. 7];
  let canonical_NaN_BF16 : bits(16) = 0x7FC0;
  match fmt {
    OFP8_E4M3 => {
      let exp  = v[6 .. 3];
      let frac = v[2 .. 0];
      if exp == 0b1111 & frac == 0b111 then canonical_NaN_BF16
      else if exp == 0b0000 then {
        // Zero or subnormal: frac * 2^-9.
        match frac {
          0b000               => sign @ zeros(15),
          0b001               => sign @ to_bits(8, 127 - 9) @ 0b0000000,
          0b01 @ f : bits(1)  => sign @ to_bits(8, 127 - 8) @ f @ 0b000000,
          0b1  @ f : bits(2)  => sign @ to_bits(8, 127 - 7) @ f @ 0b00000,
        }
      } else sign @ (zero_extend(8, exp) + to_bits(8, 127 - 7)) @ frac @ 0b0000
    },
    OFP8_E5M2 => {
      let exp  = v[6 .. 2];
      let frac = v[1 .. 0];
      if exp == 0b11111 then {
        if frac == 0b00 then sign @ 0xFF @ 0b0000000 else canonical_NaN_BF16
      } else if exp == 0b00000 then {
        // Zero or subnormal: frac * 2^-16.
        match frac {
          0b00              => sign @ zeros(15),
          0b01              => sign @ to_bits(8, 127 - 16) @ 0b0000000,
          0b1 @ f : bits(1) => sign @ to_bits(8, 127 - 15) @ f @ 0b000000,
        }
      } else sign @ (zero_extend(8, exp) + to_bits(8, 127 - 15)) @ frac @ 0b00000
    },
  }
}

// Round the non-negative integer `sig` to a multiple of 2^`shift`, returning
// the result divided by 2^`shift` and whether the result is inexact.
private function ofp8_round_shift(rm : rounding_mode, sign : bits(1), sig : nat, shift : int) -> (nat, bool) = {
  if shift <= 0 then (sig * 2 ^ (0 - shift), false)
  else {
    let divisor   = 2 ^ shift;
    let truncated = sig / divisor;
    let remainder = sig % divisor;
    let inexact   = remainder != 0;
    let round_up : bool = match rm {
      RM_RNE => 2 * remainder > divisor | (2 * remainder == divisor & truncated % 2 == 1),
      RM_RMM => 2 * remainder >= divisor,
      RM_RTZ => false,
      RM_RDN => inexact & sign == 0b1,
      RM_RUP => inexact & sign == 0b0,
      RM_DYN => internal_error(__FILE__, __LINE__, "Unresolved dynamic rounding mode"),
    };
    assert(truncated >= 0);
    (if round_up then truncated + 1 else truncated, inexact)
  }
}

// Index of the most-significant set bit.
private function ofp8_msb(sig : bits(24)) -> range(0, 23) = {
  foreach (i from 23 downto 1) {
    if sig[i] == bitone then return i;
  };
  0
}

// Convert an FP32 value to OFP8, rounding according to `rm`.
//
// Following the OFP8 specification, values that cannot be represented are
// handled as follows:
//
// * NaNs are converted to the canonical NaN.
// * In saturating mode, infinities and results that overflow are converted to
//   the maximum-magnitude finite value of the same sign.
// * In non-saturating mode infinite results are representable in E5M2, but
//   not in E4M3, where they are converted to the canonical NaN. Note that
//   overflow only produces an infinite result if the rounding mode rounds
//   away from zero for the sign in question (as in IEEE 754); otherwise the
//   result is the maximum-magnitude finite value.
//
// Overflow, underflow (with tininess detected after rounding) and inexact
// are reported as for other floating-point conversions. Converting an
// infinity is exact and does not raise any flags.
function f32_to_ofp8(fmt : ofp8_format, saturate : bool, rm : rounding_mode, v : bits(32)) -> (bits_fflags, bits(8)) = {
  let (sign, exp, mant) = fsplit_S(v);
  let max_finite = sign @ ofp8_max_finite(fmt);
  let infinity : bits(8) = match fmt {
    OFP8_E4M3 => ofp8_canonical_NaN(),
    OFP8_E5M2 => sign @ 0b1111100,
  };

  if exp == ones() then {
    return if mant != zeros() then (if mant[22] == bitzero then nvFlag() else zeros(), ofp8_canonical_NaN())
    else (zeros(), if saturate then max_finite else infinity)
  };

  if exp == zeros() & mant == zeros() then return (zeros(), sign @ 0b0000000);

  // The value is sig * 2^sig_exp. E is the exponent of its leading bit.
  let sig_bits : bits(24) = (if exp == zeros() then 0b0 else 0b1) @ mant;
  let sig      = unsigned(sig_bits);
  let sig_exp  = if exp == zeros() then -149 else unsigned(exp) - 150;
  let msb      = ofp8_msb(sig_bits);
  let E        = sig_exp + msb;

  let m    = ofp8_frac_bits(fmt);
  let bias = ofp8_bias(fmt);
  let emin = 1 - bias;

  // Round to m fraction bits assuming an unbounded exponent range. The
  // result u is in the range 2^m ..= 2^(m+1).
  let (u, inexact_u) = ofp8_round_shift(rm, sign, sig, msb - m);
  let (u, E_u) : (nat, int) = if u == 2 ^ (m + 1) then (2 ^ m, E + 1) else (u, E);
  let biased_E_u = E_u + bias;

  // The encoding of finite values is monotonic in their magnitude, so
  // overflow can be detected by comparing against the largest encoding.
  if biased_E_u * 2 ^ m + (u - 2 ^ m) > unsigned(ofp8_max_finite(fmt)) then {
    let toward_zero : bool = match rm {
      RM_RTZ => true,
      RM_RDN => sign == 0b0,
      RM_RUP => sign == 0b1,
      _      => false,
    };
    return (ofFlag() | nxFlag(), if saturate | toward_zero then max_finite else infinity)
  };

  if E >= emin then {
    // Normal result.
    let code : bits(7) = to_bits_checked(biased_E_u * 2 ^ m + (u - 2 ^ m));
    (if inexact_u then nxFlag() else zeros(), sign @ code)
  } else {
    // Subnormal result (which may round up to the smallest normal value):
    // round to a multiple of 2^(emin - m).
    let (s, inexact_s) = ofp8_round_shift(rm, sign, sig, (emin - m) - sig_exp);
    let code : bits(7) = to_bits_checked(s);
    // Tininess is detected after rounding: the result is tiny if rounding it
    // with an unbounded exponent range gives a magnitude below 2^emin.
    let tiny = E_u < emin;
    let fflags : bits_fflags =
      if not(inexact_s) then zeros()
      else if tiny then ufFlag() | nxFlag()
      else nxFlag();
    (fflags, sign @ code)
  }
}

// Convert a BF16 value to OFP8. BF16 to FP32 conversion is exact.
function bf16_to_ofp8(fmt : ofp8_format, saturate : bool, rm : rounding_mode, v : bits(16)) -> (bits_fflags, bits(8)) = {
  let (fflags_widen, v32) = bf16_to_f32(v);
  let (fflags, result) = f32_to_ofp8(fmt, saturate, rm, v32);
  (fflags_widen | fflags, result)
}
