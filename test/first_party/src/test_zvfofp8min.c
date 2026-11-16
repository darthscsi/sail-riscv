// Test of the Zvfofp8min extension (OFP8 conversions), version 0.9.
//
// The expected results of the conversions were produced by an independent
// model using exact rational arithmetic.

#include "common/encoding.h"
#include "common/runtime.h"

// The instructions are emitted as raw encodings and vtype is set with vsetvl
// so that no assembler support for Zvfofp8min is needed.

// vtype encodings.
#define VTYPE_E8 0x00UL
#define VTYPE_E16 0x08UL
#define VTYPE_E32 0x10UL
#define VTYPE_MF4 0x06UL
#define VTYPE_M1 0x00UL
#define VTYPE_M2 0x01UL
#define VTYPE_M4 0x02UL
#define VTYPE_M8 0x03UL
#define VTYPE_ALTFMT 0x100UL
#define VTYPE_VILL (1UL << (__riscv_xlen - 1))

// OPFVV VFUNARY0 (funct6=010010) encodings; the vs1 field selects the operation.
#define VFUNARY0(vs1, vd, vs2, vm) (0x48001057 | ((vm) << 25) | ((vs2) << 20) | ((vs1) << 15) | ((vd) << 7))
#define VS1_WCVTBF16 13     // vfwcvtbf16.f.f.v
#define VS1_NCVTBF16 29     // vfncvtbf16.f.f.w
#define VS1_NCVTBF16_SAT 31 // vfncvtbf16.sat.f.f.w
#define VS1_NCVT_Q 25       // vfncvt.f.f.q
#define VS1_NCVT_SAT_Q 27   // vfncvt.sat.f.f.q

enum {
  WCVTBF16,
  NCVTBF16,
  NCVTBF16_SAT,
  NCVT_Q,
  NCVT_SAT_Q
};
enum {
  RNE,
  RTZ,
  RDN,
  RUP,
  RMM
};
enum {
  NX = 1,
  UF = 2,
  OF = 4,
  NV = 16
};

#define STR_(x) #x
#define STR(x) STR_(x)

// Convert a single element with vd=v4, vs2=v8 and return the result and the
// accrued exception flags. The source is written as a 32-bit element, of
// which BF16 and OFP8 sources use the low bits.
#define DEFINE_CONVERT(name, vs1, result_vtype, result_mask)                                                           \
  static __attribute__((noinline)) uint32_t                                                                            \
  name(unsigned long altfmt, unsigned long frm, unsigned long input, unsigned long *fflags) {                          \
    unsigned long result, flags, tmp;                                                                                  \
    __asm__ volatile(                                                                                                  \
      "li %[tmp], 1\n"                                                                                                 \
      "li %[res], 0x12\n" /* e32, m4 */                                                                                \
      "vsetvl x0, %[tmp], %[res]\n"                                                                                    \
      "vmv.v.i v4, -1\n"                                                                                               \
      "vmv.s.x v8, %[in]\n"                                                                                            \
      "slli %[res], %[alt], 8\n" /* e8, m1, vtype.altfmt = altfmt */                                                   \
      "vsetvl x0, %[tmp], %[res]\n"                                                                                    \
      "fsrm %[frm]\n"                                                                                                  \
      "fsflags x0\n"                                                                                                   \
      ".4byte " STR(VFUNARY0(vs1, 4, 8, 1)) "\n"                                                                       \
                                            "frflags %[ff]\n"                                                          \
                                            "fsrm x0\n"                                                                \
                                            "li %[res], " STR(result_vtype) "\n"                                       \
                                                                            "vsetvl x0, %[tmp], %[res]\n"              \
                                                                            "vmv.x.s %[res], v4\n"                     \
      : [res] "=&r"(result), [ff] "=&r"(flags), [tmp] "=&r"(tmp)                                                       \
      : [alt] "r"(altfmt), [frm] "r"(frm), [in] "r"(input)                                                             \
    );                                                                                                                 \
    *fflags = flags;                                                                                                   \
    return (uint32_t)result & result_mask;                                                                             \
  }

DEFINE_CONVERT(vfwcvtbf16_f_f_v, VS1_WCVTBF16, 0x09 /* e16, m2 */, 0xFFFF)
DEFINE_CONVERT(vfncvtbf16_f_f_w, VS1_NCVTBF16, 0x00 /* e8, m1 */, 0xFF)
DEFINE_CONVERT(vfncvtbf16_sat_f_f_w, VS1_NCVTBF16_SAT, 0x00, 0xFF)
DEFINE_CONVERT(vfncvt_f_f_q, VS1_NCVT_Q, 0x00, 0xFF)
DEFINE_CONVERT(vfncvt_sat_f_f_q, VS1_NCVT_SAT_Q, 0x00, 0xFF)

static uint32_t convert(unsigned kind, unsigned altfmt, unsigned frm, uint32_t input, unsigned long *fflags) {
  switch (kind) {
  case WCVTBF16:
    return vfwcvtbf16_f_f_v(altfmt, frm, input, fflags);
  case NCVTBF16:
    return vfncvtbf16_f_f_w(altfmt, frm, input, fflags);
  case NCVTBF16_SAT:
    return vfncvtbf16_sat_f_f_w(altfmt, frm, input, fflags);
  case NCVT_Q:
    return vfncvt_f_f_q(altfmt, frm, input, fflags);
  default:
    return vfncvt_sat_f_f_q(altfmt, frm, input, fflags);
  }
}

// vtype.altfmt selects the OFP8 format: 0 is E4M3, 1 is E5M2.
static const struct {
  uint8_t kind, altfmt, frm;
  uint32_t input;
  uint16_t expected;
  uint8_t fflags;
} conversions[] = {
  // OFP8 E4M3 -> BF16: exact, never raises flags.
  {WCVTBF16, 0, RNE, 0x00, 0x0000, 0}, // +0
  {WCVTBF16, 0, RNE, 0x80, 0x8000, 0}, // -0
  {WCVTBF16, 0, RNE, 0x01, 0x3b00, 0}, // smallest subnormal, 2^-9
  {WCVTBF16, 0, RNE, 0x07, 0x3c60, 0}, // largest subnormal, 7 * 2^-9
  {WCVTBF16, 0, RNE, 0x08, 0x3c80, 0}, // smallest normal, 2^-6
  {WCVTBF16, 0, RNE, 0x38, 0x3f80, 0}, // 1.0
  {WCVTBF16, 0, RNE, 0xb8, 0xbf80, 0}, // -1.0
  {WCVTBF16, 0, RNE, 0x7e, 0x43e0, 0}, // largest finite, 448
  {WCVTBF16, 0, RNE, 0xfe, 0xc3e0, 0}, // -448
  {WCVTBF16, 0, RNE, 0x7f, 0x7fc0, 0}, // NaN
  {WCVTBF16, 0, RNE, 0xff, 0x7fc0, 0}, // NaN with sign bit set

  // OFP8 E5M2 -> BF16.
  {WCVTBF16, 1, RNE, 0x00, 0x0000, 0}, // +0
  {WCVTBF16, 1, RNE, 0x01, 0x3780, 0}, // smallest subnormal, 2^-16
  {WCVTBF16, 1, RNE, 0x03, 0x3840, 0}, // largest subnormal, 3 * 2^-16
  {WCVTBF16, 1, RNE, 0x04, 0x3880, 0}, // smallest normal, 2^-14
  {WCVTBF16, 1, RNE, 0x3c, 0x3f80, 0}, // 1.0
  {WCVTBF16, 1, RNE, 0x7b, 0x4760, 0}, // largest finite, 57344
  {WCVTBF16, 1, RNE, 0x7c, 0x7f80, 0}, // +infinity
  {WCVTBF16, 1, RNE, 0xfc, 0xff80, 0}, // -infinity
  {WCVTBF16, 1, RNE, 0x7d, 0x7fc0, 0}, // signaling NaN: no invalid flag
  {WCVTBF16, 1, RNE, 0x7e, 0x7fc0, 0}, // quiet NaN
  {WCVTBF16, 1, RNE, 0xff, 0x7fc0, 0}, // NaN with sign bit set

  // FP32 -> OFP8 E4M3: special values.
  {NCVT_Q, 0, RNE, 0x3f800000, 0x38, 0},     // 1.0
  {NCVT_Q, 0, RNE, 0xbf800000, 0xb8, 0},     // -1.0
  {NCVT_Q, 0, RNE, 0x80000000, 0x80, 0},     // -0
  {NCVT_Q, 0, RNE, 0x7fc00000, 0x7f, 0},     // quiet NaN -> canonical NaN
  {NCVT_Q, 0, RNE, 0xffc00000, 0x7f, 0},     // negative quiet NaN -> canonical NaN
  {NCVT_Q, 0, RNE, 0x7f800001, 0x7f, NV},    // signaling NaN -> canonical NaN, invalid
  {NCVT_Q, 0, RNE, 0x7f800000, 0x7f, 0},     // +infinity -> canonical NaN, no flags
  {NCVT_Q, 0, RNE, 0xff800000, 0x7f, 0},     // -infinity -> canonical NaN, no flags
  {NCVT_SAT_Q, 0, RNE, 0x7f800000, 0x7e, 0}, // saturating: +infinity -> largest finite, no flags
  {NCVT_SAT_Q, 0, RNE, 0xff800000, 0xfe, 0}, // saturating: -infinity -> most negative finite, no flags
  {NCVT_SAT_Q, 0, RNE, 0x7fc00000, 0x7f, 0}, // saturating: NaN is still NaN

  // FP32 -> OFP8 E4M3: overflow. Largest finite value is 448; 464 is halfway to the next value.
  {NCVT_Q, 0, RNE, 0x43e00000, 0x7e, 0},           // largest finite value, exact
  {NCVT_Q, 0, RNE, 0x43e7ffff, 0x7e, NX},          // just below halfway: rounds down, inexact only
  {NCVT_Q, 0, RNE, 0x43e80000, 0x7e, NX},          // halfway: ties to even rounds down
  {NCVT_Q, 0, RMM, 0x43e80000, 0x7f, OF | NX},     // halfway, RMM: overflow -> canonical NaN
  {NCVT_Q, 0, RNE, 0x43e88000, 0x7f, OF | NX},     // just above halfway: overflow
  {NCVT_Q, 0, RTZ, 0x43e88000, 0x7e, NX},          // RTZ rounds down: not an overflow
  {NCVT_Q, 0, RTZ, 0x43f00000, 0x7e, OF | NX},     // RTZ overflow -> largest finite
  {NCVT_Q, 0, RUP, 0x43f00000, 0x7f, OF | NX},     // RUP overflow of a positive value -> canonical NaN
  {NCVT_Q, 0, RDN, 0x43f00000, 0x7e, OF | NX},     // RDN overflow of a positive value -> largest finite
  {NCVT_Q, 0, RDN, 0xc3f00000, 0x7f, OF | NX},     // RDN overflow of a negative value -> canonical NaN
  {NCVT_Q, 0, RUP, 0xc3f00000, 0xfe, OF | NX},     // RUP overflow of a negative value -> most negative finite
  {NCVT_Q, 0, RNE, 0x7f7fffff, 0x7f, OF | NX},     // FLT_MAX
  {NCVT_SAT_Q, 0, RNE, 0x43e88000, 0x7e, OF | NX}, // saturating overflow -> largest finite, still raises overflow
  {NCVT_SAT_Q, 0, RUP, 0xc3f00000, 0xfe, OF | NX}, // saturating overflow of a negative value
  {NCVT_SAT_Q, 0, RNE, 0xff7fffff, 0xfe, OF | NX}, // saturating: -FLT_MAX
  {NCVT_SAT_Q, 0, RNE, 0x43e7ffff, 0x7e, NX},      // saturating: no overflow, same as non-saturating

  // FP32 -> OFP8 E4M3: underflow. Smallest subnormal is 0.00195312, smallest normal is 0.015625.
  {NCVT_Q, 0, RNE, 0x3b000000, 0x01, 0},       // smallest subnormal, exact: no underflow flag
  {NCVT_Q, 0, RNE, 0x3b400000, 0x02, UF | NX}, // halfway between subnormals: ties to even
  {NCVT_Q, 0, RNE, 0x3a800000, 0x00, UF | NX}, // half the smallest subnormal: ties to even gives zero
  {NCVT_Q, 0, RNE, 0x3a800001, 0x01, UF | NX}, // just above half the smallest subnormal
  {NCVT_Q, 0, RUP, 0x00000001, 0x01, UF | NX}, // smallest FP32 subnormal, RUP
  {NCVT_Q, 0, RDN, 0x80000001, 0x81, UF | NX}, // negative smallest FP32 subnormal, RDN
  {NCVT_Q, 0, RTZ, 0x00000001, 0x00, UF | NX}, // smallest FP32 subnormal, RTZ
  {NCVT_Q, 0, RNE, 0x3c780000, 0x08, NX},      // rounds to the smallest normal even with unbounded exponent: not tiny
  {NCVT_Q, 0, RNE, 0x3c77ffff, 0x08, UF | NX}, // rounds to the smallest normal only because it is subnormal: tiny
  {NCVT_Q, 0, RTZ, 0x3c780000, 0x07, UF | NX}, // largest subnormal, inexact: underflow
  {NCVT_Q, 0, RNE, 0x3c800000, 0x08, 0},       // smallest normal, exact

  // BF16 -> OFP8 E4M3.
  {NCVTBF16, 0, RNE, 0x3f80, 0x38, 0},           // 1.0
  {NCVTBF16, 0, RNE, 0xbfc0, 0xbc, 0},           // -1.5
  {NCVTBF16, 0, RNE, 0x7fc0, 0x7f, 0},           // quiet NaN
  {NCVTBF16, 0, RNE, 0x7f81, 0x7f, NV},          // signaling NaN: invalid
  {NCVTBF16, 0, RNE, 0x7f80, 0x7f, 0},           // +infinity
  {NCVTBF16, 0, RNE, 0xff80, 0x7f, 0},           // -infinity
  {NCVTBF16_SAT, 0, RNE, 0x7f80, 0x7e, 0},       // saturating: +infinity
  {NCVTBF16_SAT, 0, RNE, 0xff80, 0xfe, 0},       // saturating: -infinity
  {NCVTBF16, 0, RNE, 0x43e8, 0x7e, NX},          // halfway between largest finite and the next value
  {NCVTBF16, 0, RMM, 0x43e8, 0x7f, OF | NX},     // same, RMM
  {NCVTBF16, 0, RNE, 0x43f0, 0x7f, OF | NX},     // overflow
  {NCVTBF16, 0, RTZ, 0x43f0, 0x7e, OF | NX},     // overflow, RTZ
  {NCVTBF16_SAT, 0, RNE, 0x43f0, 0x7e, OF | NX}, // saturating overflow
  {NCVTBF16_SAT, 0, RDN, 0xc3f0, 0xfe, OF | NX}, // saturating overflow of a negative value
  {NCVTBF16, 0, RNE, 0x3f88, 0x38, NX},          // 1.0 plus half an ulp: ties to even rounds down
  {NCVTBF16, 0, RNE, 0x3f98, 0x3a, NX},          // 1.0 plus one and a half ulp: ties to even rounds up
  {NCVTBF16, 0, RUP, 0x3f81, 0x39, NX},          // RUP
  {NCVTBF16, 0, RDN, 0xbf81, 0xb9, NX},          // RDN
  {NCVTBF16, 0, RNE, 0x3a80, 0x00, UF | NX},     // half the smallest subnormal
  {NCVTBF16, 0, RUP, 0x0001, 0x01, UF | NX},     // smallest BF16 subnormal, RUP
  {NCVTBF16, 0, RNE, 0x3c78, 0x08, NX},          // rounds up to the smallest normal: not tiny

  // FP32 -> OFP8 E5M2: special values.
  {NCVT_Q, 1, RNE, 0x3f800000, 0x3c, 0},     // 1.0
  {NCVT_Q, 1, RNE, 0xbf800000, 0xbc, 0},     // -1.0
  {NCVT_Q, 1, RNE, 0x80000000, 0x80, 0},     // -0
  {NCVT_Q, 1, RNE, 0x7fc00000, 0x7f, 0},     // quiet NaN -> canonical NaN
  {NCVT_Q, 1, RNE, 0xffc00000, 0x7f, 0},     // negative quiet NaN -> canonical NaN
  {NCVT_Q, 1, RNE, 0x7f800001, 0x7f, NV},    // signaling NaN -> canonical NaN, invalid
  {NCVT_Q, 1, RNE, 0x7f800000, 0x7c, 0},     // +infinity
  {NCVT_Q, 1, RNE, 0xff800000, 0xfc, 0},     // -infinity
  {NCVT_SAT_Q, 1, RNE, 0x7f800000, 0x7b, 0}, // saturating: +infinity -> largest finite, no flags
  {NCVT_SAT_Q, 1, RNE, 0xff800000, 0xfb, 0}, // saturating: -infinity -> most negative finite, no flags
  {NCVT_SAT_Q, 1, RNE, 0x7fc00000, 0x7f, 0}, // saturating: NaN is still NaN

  // FP32 -> OFP8 E5M2: overflow. Largest finite value is 57344; 61440 is halfway to the next value.
  {NCVT_Q, 1, RNE, 0x47600000, 0x7b, 0},           // largest finite value, exact
  {NCVT_Q, 1, RNE, 0x476fffff, 0x7b, NX},          // just below halfway: rounds down, inexact only
  {NCVT_Q, 1, RNE, 0x47700000, 0x7c, OF | NX},     // halfway: ties to even rounds up: overflow to infinity
  {NCVT_Q, 1, RMM, 0x47700000, 0x7c, OF | NX},     // halfway, RMM: overflow -> infinity
  {NCVT_Q, 1, RNE, 0x47700100, 0x7c, OF | NX},     // just above halfway: overflow
  {NCVT_Q, 1, RTZ, 0x47700100, 0x7b, NX},          // RTZ rounds down: not an overflow
  {NCVT_Q, 1, RTZ, 0x47800000, 0x7b, OF | NX},     // RTZ overflow -> largest finite
  {NCVT_Q, 1, RUP, 0x47800000, 0x7c, OF | NX},     // RUP overflow of a positive value -> infinity
  {NCVT_Q, 1, RDN, 0x47800000, 0x7b, OF | NX},     // RDN overflow of a positive value -> largest finite
  {NCVT_Q, 1, RDN, 0xc7800000, 0xfc, OF | NX},     // RDN overflow of a negative value -> -infinity
  {NCVT_Q, 1, RUP, 0xc7800000, 0xfb, OF | NX},     // RUP overflow of a negative value -> most negative finite
  {NCVT_Q, 1, RNE, 0x7f7fffff, 0x7c, OF | NX},     // FLT_MAX
  {NCVT_SAT_Q, 1, RNE, 0x47700100, 0x7b, OF | NX}, // saturating overflow -> largest finite, still raises overflow
  {NCVT_SAT_Q, 1, RUP, 0xc7800000, 0xfb, OF | NX}, // saturating overflow of a negative value
  {NCVT_SAT_Q, 1, RNE, 0xff7fffff, 0xfb, OF | NX}, // saturating: -FLT_MAX
  {NCVT_SAT_Q, 1, RNE, 0x476fffff, 0x7b, NX},      // saturating: no overflow, same as non-saturating

  // FP32 -> OFP8 E5M2: underflow. Smallest subnormal is 1.52588e-05, smallest normal is 6.10352e-05.
  {NCVT_Q, 1, RNE, 0x37800000, 0x01, 0},       // smallest subnormal, exact: no underflow flag
  {NCVT_Q, 1, RNE, 0x37c00000, 0x02, UF | NX}, // halfway between subnormals: ties to even
  {NCVT_Q, 1, RNE, 0x37000000, 0x00, UF | NX}, // half the smallest subnormal: ties to even gives zero
  {NCVT_Q, 1, RNE, 0x37000001, 0x01, UF | NX}, // just above half the smallest subnormal
  {NCVT_Q, 1, RUP, 0x00000001, 0x01, UF | NX}, // smallest FP32 subnormal, RUP
  {NCVT_Q, 1, RDN, 0x80000001, 0x81, UF | NX}, // negative smallest FP32 subnormal, RDN
  {NCVT_Q, 1, RTZ, 0x00000001, 0x00, UF | NX}, // smallest FP32 subnormal, RTZ
  {NCVT_Q, 1, RNE, 0x38700000, 0x04, NX},      // rounds to the smallest normal even with unbounded exponent: not tiny
  {NCVT_Q, 1, RNE, 0x386fffff, 0x04, UF | NX}, // rounds to the smallest normal only because it is subnormal: tiny
  {NCVT_Q, 1, RTZ, 0x38700000, 0x03, UF | NX}, // largest subnormal, inexact: underflow
  {NCVT_Q, 1, RNE, 0x38800000, 0x04, 0},       // smallest normal, exact

  // BF16 -> OFP8 E5M2.
  {NCVTBF16, 1, RNE, 0x3f80, 0x3c, 0},           // 1.0
  {NCVTBF16, 1, RNE, 0xbfc0, 0xbe, 0},           // -1.5
  {NCVTBF16, 1, RNE, 0x7fc0, 0x7f, 0},           // quiet NaN
  {NCVTBF16, 1, RNE, 0x7f81, 0x7f, NV},          // signaling NaN: invalid
  {NCVTBF16, 1, RNE, 0x7f80, 0x7c, 0},           // +infinity
  {NCVTBF16, 1, RNE, 0xff80, 0xfc, 0},           // -infinity
  {NCVTBF16_SAT, 1, RNE, 0x7f80, 0x7b, 0},       // saturating: +infinity
  {NCVTBF16_SAT, 1, RNE, 0xff80, 0xfb, 0},       // saturating: -infinity
  {NCVTBF16, 1, RNE, 0x4770, 0x7c, OF | NX},     // halfway between largest finite and the next value
  {NCVTBF16, 1, RMM, 0x4770, 0x7c, OF | NX},     // same, RMM
  {NCVTBF16, 1, RNE, 0x4780, 0x7c, OF | NX},     // overflow
  {NCVTBF16, 1, RTZ, 0x4780, 0x7b, OF | NX},     // overflow, RTZ
  {NCVTBF16_SAT, 1, RNE, 0x4780, 0x7b, OF | NX}, // saturating overflow
  {NCVTBF16_SAT, 1, RDN, 0xc780, 0xfb, OF | NX}, // saturating overflow of a negative value
  {NCVTBF16, 1, RNE, 0x3f90, 0x3c, NX},          // 1.0 plus half an ulp: ties to even rounds down
  {NCVTBF16, 1, RNE, 0x3fb0, 0x3e, NX},          // 1.0 plus one and a half ulp: ties to even rounds up
  {NCVTBF16, 1, RUP, 0x3f81, 0x3d, NX},          // RUP
  {NCVTBF16, 1, RDN, 0xbf81, 0xbd, NX},          // RDN
  {NCVTBF16, 1, RNE, 0x3700, 0x00, UF | NX},     // half the smallest subnormal
  {NCVTBF16, 1, RUP, 0x0001, 0x01, UF | NX},     // smallest BF16 subnormal, RUP
  {NCVTBF16, 1, RNE, 0x3870, 0x04, NX},          // rounds up to the smallest normal: not tiny
};

// ---------------------------------------------------------------------------
// Checks of illegal instructions and of vtype.

#if __riscv_xlen == 64
#define STORE_XLEN sd
#else
#define STORE_XLEN sw
#endif

// Illegal instruction trap handler: record the trap and skip the instruction.
// Uses t5 and t6, which are clobbered by run_insn().
volatile unsigned long trap_cause;
__asm__(".pushsection .text\n"
        ".p2align 6\n"
        "skip_trap_handler:\n"
        "  csrr t5, mcause\n"
        "  la t6, trap_cause\n"
        "  " STR(STORE_XLEN) " t5, 0(t6)\n"
                             "  csrr t5, mepc\n"
                             "  addi t5, t5, 4\n"
                             "  csrw mepc, t5\n"
                             "  mret\n"
                             ".popsection\n");
extern void skip_trap_handler(void);

// A single instruction followed by ret.
static uint32_t insn_buffer[2] __attribute__((aligned(4)));

#define NO_TRAP (~0UL)
#define ILLEGAL_INSTRUCTION 2UL

// Execute `insn` after setting vl=1, vtype and frm. Returns mcause if the
// instruction trapped, otherwise NO_TRAP. vtype afterwards is written to
// `vtype_after`.
static __attribute__((noinline)) unsigned long run_insn(
  uint32_t insn,
  unsigned long vtype,
  unsigned long frm,
  unsigned long *vtype_after
) {
  unsigned long tmp;
  insn_buffer[0] = insn;
  insn_buffer[1] = 0x00008067; // ret
  trap_cause = NO_TRAP;
  __asm__ volatile("fence.i\n"
                   "li t1, 1\n"
                   "vsetvl x0, t1, %[vtype]\n"
                   "fsrm %[frm]\n"
                   "jalr ra, 0(%[buf])\n"
                   "csrr %[tmp], vtype\n"
                   "fsrm x0\n"
                   : [tmp] "=&r"(tmp)
                   : [vtype] "r"(vtype), [frm] "r"(frm), [buf] "r"(insn_buffer)
                   : "ra", "t0", "t1", "t5", "t6", "memory");
  *vtype_after = tmp;
  return trap_cause;
}

static int failures;

static void check_trap(int line, int expect_trap, uint32_t insn, unsigned long vtype, unsigned long frm) {
  unsigned long vtype_after;
  unsigned long cause = run_insn(insn, vtype, frm, &vtype_after);
  if (cause != (expect_trap ? ILLEGAL_INSTRUCTION : NO_TRAP)) {
    printf(
      "line %d: insn 0x%08x vtype 0x%lx frm %lu: mcause %ld, expected %s\n",
      line,
      (unsigned)insn,
      vtype,
      frm,
      (long)cause,
      expect_trap ? "an illegal instruction trap" : "no trap"
    );
    failures++;
  }
}
#define LEGAL(insn, vtype) check_trap(__LINE__, 0, insn, vtype, RNE)
#define ILLEGAL(insn, vtype) check_trap(__LINE__, 1, insn, vtype, RNE)

// vsetvli t0, t1, zimm and vsetivli t0, 1, zimm. t1 is 1 in run_insn().
#define VSETVLI(zimm) (((uint32_t)(zimm) << 20) | (6u << 15) | (7u << 12) | (5u << 7) | 0x57u)
#define VSETIVLI(zimm) (0xC0000000u | ((uint32_t)(zimm) << 20) | (1u << 15) | (7u << 12) | (5u << 7) | 0x57u)

static void check_vtype(int line, uint32_t insn, unsigned long vtype, unsigned long expected) {
  unsigned long vtype_after;
  unsigned long cause = run_insn(insn, vtype, RNE, &vtype_after);
  if (cause != NO_TRAP || vtype_after != expected) {
    printf(
      "line %d: insn 0x%08x vtype 0x%lx: mcause %ld vtype 0x%lx, expected vtype 0x%lx\n",
      line,
      (unsigned)insn,
      vtype,
      (long)cause,
      vtype_after,
      expected
    );
    failures++;
  }
}
#define NOP 0x00000013u

static void test_conversions(void) {
  for (unsigned i = 0; i < sizeof(conversions) / sizeof(conversions[0]); i++) {
    unsigned long fflags;
    uint32_t result =
      convert(conversions[i].kind, conversions[i].altfmt, conversions[i].frm, conversions[i].input, &fflags);
    if (result != conversions[i].expected || fflags != conversions[i].fflags) {
      printf(
        "conversion %u: kind %u altfmt %u frm %u input 0x%08x: got 0x%x fflags 0x%lx, expected 0x%x fflags 0x%x\n",
        i,
        conversions[i].kind,
        conversions[i].altfmt,
        conversions[i].frm,
        (unsigned)conversions[i].input,
        (unsigned)result,
        fflags,
        conversions[i].expected,
        conversions[i].fflags
      );
      failures++;
    }
  }
}

static void test_vtype(void) {
  // vtype.altfmt is supported when SEW=8, by vsetvl, vsetvli and vsetivli.
  check_vtype(__LINE__, NOP, VTYPE_E8 | VTYPE_ALTFMT, VTYPE_E8 | VTYPE_ALTFMT);
  check_vtype(__LINE__, NOP, VTYPE_E8 | VTYPE_M2 | VTYPE_ALTFMT | 0xC0, VTYPE_E8 | VTYPE_M2 | VTYPE_ALTFMT | 0xC0);
  check_vtype(__LINE__, VSETVLI(VTYPE_E8 | VTYPE_ALTFMT), VTYPE_E8, VTYPE_E8 | VTYPE_ALTFMT);
  check_vtype(__LINE__, VSETIVLI(VTYPE_E8 | VTYPE_ALTFMT), VTYPE_E8, VTYPE_E8 | VTYPE_ALTFMT);
  check_vtype(__LINE__, VSETVLI(VTYPE_E8), VTYPE_E8 | VTYPE_ALTFMT, VTYPE_E8);
  // Zvfofp8min does not define altfmt=1 for other element widths.
  check_vtype(__LINE__, NOP, VTYPE_E16 | VTYPE_ALTFMT, VTYPE_VILL);
  check_vtype(__LINE__, NOP, VTYPE_E32 | VTYPE_ALTFMT, VTYPE_VILL);
  check_vtype(__LINE__, VSETVLI(VTYPE_E16 | VTYPE_ALTFMT), VTYPE_E8, VTYPE_VILL);
  check_vtype(__LINE__, VSETIVLI(VTYPE_E32 | VTYPE_ALTFMT), VTYPE_E8, VTYPE_VILL);
  // The remaining vtype bits are still reserved.
  check_vtype(__LINE__, NOP, VTYPE_E8 | 0x200, VTYPE_VILL);
  check_vtype(__LINE__, VSETVLI(VTYPE_E8 | VTYPE_ALTFMT | 0x200), VTYPE_E8, VTYPE_VILL);
}

static void test_illegal(void) {
  static const uint8_t ops[] = {VS1_WCVTBF16, VS1_NCVTBF16, VS1_NCVTBF16_SAT, VS1_NCVT_Q, VS1_NCVT_SAT_Q};

  for (unsigned altfmt = 0; altfmt <= VTYPE_ALTFMT; altfmt += VTYPE_ALTFMT) {
    for (unsigned i = 0; i < sizeof(ops); i++) {
      // Legal with SEW=8, but an invalid dynamic rounding mode is reserved.
      LEGAL(VFUNARY0(ops[i], 4, 8, 1), VTYPE_E8 | altfmt);
      check_trap(__LINE__, 1, VFUNARY0(ops[i], 4, 8, 1), VTYPE_E8 | altfmt, 5);
      // A masked instruction cannot write v0.
      LEGAL(VFUNARY0(ops[i], 4, 8, 0), VTYPE_E8 | altfmt);
      ILLEGAL(VFUNARY0(ops[i], 0, 8, 0), VTYPE_E8 | altfmt);
    }
  }

  // With SEW=16 the two BF16 conversions are the Zvfbfmin instructions. The
  // new encodings are reserved.
  LEGAL(VFUNARY0(VS1_WCVTBF16, 4, 8, 1), VTYPE_E16);
  LEGAL(VFUNARY0(VS1_NCVTBF16, 4, 8, 1), VTYPE_E16);
  ILLEGAL(VFUNARY0(VS1_NCVTBF16_SAT, 4, 8, 1), VTYPE_E16);
  ILLEGAL(VFUNARY0(VS1_NCVT_Q, 4, 8, 1), VTYPE_E16);
  ILLEGAL(VFUNARY0(VS1_NCVT_SAT_Q, 4, 8, 1), VTYPE_E16);
  for (unsigned i = 0; i < sizeof(ops); i++) {
    ILLEGAL(VFUNARY0(ops[i], 4, 8, 1), VTYPE_E32);
  }

  // vfwcvtbf16.f.f.v: the destination has EMUL = 2 * LMUL.
  LEGAL(VFUNARY0(VS1_WCVTBF16, 8, 16, 1), VTYPE_E8 | VTYPE_M4);
  ILLEGAL(VFUNARY0(VS1_WCVTBF16, 8, 16, 1), VTYPE_E8 | VTYPE_M8);
  ILLEGAL(VFUNARY0(VS1_WCVTBF16, 5, 8, 1), VTYPE_E8); // vd is not aligned
  LEGAL(VFUNARY0(VS1_WCVTBF16, 8, 9, 1), VTYPE_E8);   // overlap with the upper half of vd
  ILLEGAL(VFUNARY0(VS1_WCVTBF16, 8, 8, 1), VTYPE_E8); // overlap with the lower half of vd

  // vfncvtbf16[.sat].f.f.w: the source has EMUL = 2 * LMUL.
  for (unsigned vs1 = VS1_NCVTBF16; vs1 <= VS1_NCVTBF16_SAT; vs1 += 2) {
    LEGAL(VFUNARY0(vs1, 4, 8, 1), VTYPE_E8 | VTYPE_M4);
    ILLEGAL(VFUNARY0(vs1, 8, 16, 1), VTYPE_E8 | VTYPE_M8);
    ILLEGAL(VFUNARY0(vs1, 4, 9, 1), VTYPE_E8); // vs2 is not aligned
    LEGAL(VFUNARY0(vs1, 8, 8, 1), VTYPE_E8);   // overlap with the lowest part of vs2
    ILLEGAL(VFUNARY0(vs1, 9, 8, 1), VTYPE_E8); // overlap with the upper half of vs2
  }

  // vfncvt[.sat].f.f.q: the source has EMUL = 4 * LMUL.
  for (unsigned vs1 = VS1_NCVT_Q; vs1 <= VS1_NCVT_SAT_Q; vs1 += 2) {
    LEGAL(VFUNARY0(vs1, 4, 8, 1), VTYPE_E8 | VTYPE_MF4);
    LEGAL(VFUNARY0(vs1, 4, 8, 1), VTYPE_E8 | VTYPE_M2);
    ILLEGAL(VFUNARY0(vs1, 4, 8, 1), VTYPE_E8 | VTYPE_M4);
    ILLEGAL(VFUNARY0(vs1, 8, 16, 1), VTYPE_E8 | VTYPE_M8);
    ILLEGAL(VFUNARY0(vs1, 4, 10, 1), VTYPE_E8); // vs2 is not aligned
    LEGAL(VFUNARY0(vs1, 8, 8, 1), VTYPE_E8);    // overlap with the lowest part of vs2
    ILLEGAL(VFUNARY0(vs1, 9, 8, 1), VTYPE_E8);
    ILLEGAL(VFUNARY0(vs1, 11, 8, 1), VTYPE_E8);
  }

  for (unsigned i = 0; i < sizeof(ops); i++) {
    // vtype.vill is set.
    ILLEGAL(VFUNARY0(ops[i], 4, 8, 1), VTYPE_E16 | VTYPE_ALTFMT);
  }
}

// Conversions use the mask, the tail policy and vl like any other
// instruction.
static void test_masking(void) {
  static const uint32_t source[8] = {
    0x3f800000,
    0x40000000,
    0x40400000,
    0x40800000, // 1, 2, 3, 4
    0x40a00000,
    0x40c00000,
    0x40e00000,
    0x41000000
  }; // 5, 6, 7, 8
  static const uint8_t expected[8] = {0x38, 0xAA, 0x44, 0xAA, 0x4A, 0xAA, 0xAA, 0xAA};
  uint8_t result[8];
  unsigned long tmp;
  __asm__ volatile("li %[tmp], 8\n"
                   "vsetvli x0, %[tmp], e32, m4, tu, mu\n"
                   "vle32.v v8, (%[src])\n"
                   "vsetvli x0, %[tmp], e8, m1, tu, mu\n"
                   "li %[tmp], 0xAA\n"
                   "vmv.v.x v4, %[tmp]\n"
                   "li %[tmp], 0x55\n" // elements 0, 2, 4 and 6 are active
                   "vmv.s.x v0, %[tmp]\n"
                   "li %[tmp], 6\n" // vl = 6: elements 6 and 7 are in the tail
                   "vsetvli x0, %[tmp], e8, m1, tu, mu\n"
                   "fsrm x0\n"
                   ".4byte " STR(VFUNARY0(VS1_NCVT_Q, 4, 8, 0)) "\n"
                                                                "li %[tmp], 8\n"
                                                                "vsetvli x0, %[tmp], e8, m1, tu, mu\n"
                                                                "vse8.v v4, (%[dst])\n"
                   : [tmp] "=&r"(tmp)
                   : [src] "r"(source), [dst] "r"(result)
                   : "memory");
  for (unsigned i = 0; i < 8; i++) {
    if (result[i] != expected[i]) {
      printf("masked vfncvt.f.f.q element %u: got 0x%02x, expected 0x%02x\n", i, result[i], expected[i]);
      failures++;
    }
  }
}

int main(void) {
  // Enable the FPU and the vector unit, and skip illegal instructions.
  __asm__ volatile("csrs mstatus, %0" : : "r"((unsigned long)(MSTATUS_FS | MSTATUS_VS)));
  __asm__ volatile("csrw mtvec, %0" : : "r"(&skip_trap_handler));

  test_conversions();
  test_masking();
  test_vtype();
  test_illegal();

  return failures;
}
