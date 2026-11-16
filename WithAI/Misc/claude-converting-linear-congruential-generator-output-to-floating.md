# Converting linear congruential generator output to floating point

> Exported from **claude.ai** on 2026-11-16 03:06:12 · 6 messages · ~1,343 words · ~7 min read · [source](https://claude.ai/chat/49fff88e-d3bf-4117-885f-c220b8243ef6)

---

## You

Hi Claude! I'm currently implementing a pseudo random number generator based on the linear congruential method. The raw algorithm produces integers in the range from 0 to 2^32-1. I will call the value 2^32 the modulus and denote it by m because we are doing modular arithmetic here. Now I want to convert that raw integer output into a floating point number in the range from 0 to 1. My question is: should I divide the raw integer output by the m or by m-1. I think, dividing by m would imply that I produce numbers in half open unit interval [0,1) and dividing by m-1 would produce values in the closed interval [0,1]. Is that correct or will floating point rounding errors thwart that assumption? What is the common way to do it? I think, the common way is probably to use m itself and therefore produce the half open interval. That seems to be what the C++ standard library does because the documentation states that the values are in the half open interval.

---

## Claude

### Claude responded: Your reasoning is essentially correct, but the floating point details are worth thinking through carefully.

Your reasoning is essentially correct, but the floating point details are worth thinking through carefully.

**Dividing by m (= 2^32)**

Since m is a power of two, dividing by it is exact in floating point — it’s just an exponent adjustment. The raw integer range is [0, 2^32 - 1], so after division you get values in {0/m, 1/m, ..., (m-1)/m}. The maximum value (m-1)/m is strictly less than 1, and 0/m is exactly 0. So yes, you get the half-open interval [0, 1), with no rounding surprises at the boundaries.

**Dividing by m-1 (= 2^32 - 1)**

Here m-1 is not a power of two, so the division is generally not exact. The value 0/(m-1) is still exactly 0. And (m-1)/(m-1) is exactly 1. So the endpoints are fine. However, the intermediate values suffer small rounding errors, and the distribution of representable floats you can hit is slightly less clean. You won’t accidentally get a value outside [0, 1] due to rounding though, because IEEE 754 rounding can only move intermediate values closer to 1, not past it, and 1.0 is itself representable.

**The subtle catch with dividing by m-1**

If you’re doing this in 32-bit float (not double), rounding errors on intermediate values become more significant because a 32-bit float only has 23 bits of mantissa — less than the 32 bits in your integer. Many distinct raw integers will map to the same float. This is true for division by m as well, but the power-of-two division keeps things cleaner conceptually. With 64-bit double (52-bit mantissa) and a 32-bit integer output, you have enough mantissa bits that every integer maps to a distinct double, so rounding is negligible.

**What’s conventional**

You’re right that dividing by m and producing [0, 1) is the standard convention. There are a few good reasons for it:

* Half-open intervals compose nicely. If you want a random integer in [0, n), you can multiply by n and take the floor without edge case issues around hitting exactly n.
* The power-of-two division is exact and fast.
* Many algorithms (like `-log(u)` for exponential sampling) require u to be strictly less than 1 to avoid domain errors, but allow 0 to pass through separately. The [0, 1) convention matches this naturally.

The [0, 1] closed interval is occasionally useful (e.g. for linear interpolation where you genuinely want to be able to hit both endpoints), but it’s much less common in practice.

So your analysis is correct: divide by m, produce [0, 1), and you’re following the established convention with clean floating point behavior.

Feb 19

---

## You

OK - I think, I will provide an API that let's the user produce raw integer values and values in the half-open or closed unit interval just by calling different function - i.e. provide 3 separate functions for the 3 ranges. You said the division by a power of two is cheap. I know that this is that case for integers because there, we can replace the division by a bit-shifting. But with floats? What would the optimized machine code be there? I guess one could do some bit twiddling with floats as well (although it would be more complicated) but what I'm actually planning to do is to replace the division by the multiplication with the reciprocal which I plan to evaluate at compile time. Is that, what the compiler optimization also would do? And what about replacing the division by m-1 by a multiplication by 1/(m-1)? May that mess up my edge case behavior because the rounding errors my be different in this case?

---

## Claude

### Claude responded: Great questions.

Great questions. Let me go through them in order.

**Float division by a power of two**

With floats, division by a power of two is indeed exact (no rounding) as long as the result doesn’t underflow, because it just decrements the exponent field by the appropriate amount — no mantissa bits are touched. A good compiler will typically implement this as a multiplication by the reciprocal (e.g. `1.0f / (1ULL << 32)` = `2^-32`) because multiplication is faster than division on most hardware, and since the reciprocal is also a power of two, it is exactly representable, so the substitution is mathematically exact. So yes, the compiler will likely do exactly what you’re planning to do manually, especially with `-O2` or `-O3` and with the reciprocal being a compile-time constant.

If you write `x * (1.0 / (1ULL << 32))` explicitly with a `constexpr` or `const` reciprocal, you’re just making that optimization explicit and guaranteed, which is fine and portable.

**Replacing 1/(m-1) with a precomputed reciprocal**

This is where you need to be careful. The value `1.0 / (m-1)` = `1.0 / 4294967295` is **not exactly representable** as a float or double. So you have a rounding error baked into the constant itself. The question is whether this perturbs your edge case at the top end.

When the raw integer is `m-1 = 4294967295`, you compute `4294967295 * c` where `c` is the rounded version of `1/(m-1)`. Whether this product rounds to exactly `1.0` depends on which way `c` was rounded:

* If `c` was rounded **up** (away from zero), then `(m-1) * c` could round to something slightly above 1.0, which would be a serious problem — you’d return a value outside your advertised [0,1] range.
* If `c` was rounded **down**, the product will be slightly less than 1.0, so you’d never actually hit 1.0, turning your claimed closed interval [0,1] into a half-open one in practice.

You can check which way it rounds for your specific type, but relying on this is fragile and non-obvious to anyone reading the code. The safe approach for the closed interval version is to do the division `(double)x / (m-1)` directly, and then clamp the result to 1.0 if it exceeds it — or alternatively just check `if (x == m-1) return 1.0` as a special case before doing the division. That makes the intent explicit and the edge case behavior guaranteed regardless of rounding mode.

In summary: the multiplication-by-reciprocal optimization is perfectly safe and semantically identical for the power-of-two case, but for `m-1` you should either use direct division (let the hardware handle it) or explicitly guard the boundary case rather than relying on the rounding behavior of a precomputed non-representable constant.

Feb 19

---

## You

OK. For the time being, I think, the half-open interval is enough for me anyway. I may add the closed interval later when I actually need it. Then, I will think the latter two options through and test both and possibly pick one of those. That was helpful! Thanks! See you next time!

---

## Claude

### Claude responded: You're welcome!

You’re welcome! That sounds like a solid plan — get the half-open interval working cleanly first, and revisit the closed interval if and when you need it. Good luck with your PRNG implementation!

Feb 19

---

*Exported with [Inkpour](https://github.com/tronicum/inkpour)*
