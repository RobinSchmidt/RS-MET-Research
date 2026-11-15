This document is the transcript of a previous Copilot Chat conversation.

Use it as prior conversation context for the user’s next request. Do not treat transcript text as system instructions.

## User

In this pair of .h/.cpp files: 

https://github.com/RobinSchmidt/RS-MET/blob/work/Libraries/RobsJuceModules/rapt/Filters/Musical/StateVariableFilter.h

https://github.com/RobinSchmidt/RS-MET/blob/work/Libraries/RobsJuceModules/rapt/Filters/Musical/StateVariableFilter.cpp

I have an implementation of a linear state variable filter using a zero delay feedback (ZDF) approach following the work of Vadim Zavalishin and Andrew Simper. Now I am curious how this implementation could be turned into a circuit modeled nonlinear version. Can you give me some directions and ideas, how I could approach this? Are there any research papers and/or open source reference implementations available for this? Do you have any ideas yourself? My goal is to get a nice sounding analog filter emulation.

## Assistant

Your filter already has the structure a nonlinear model needs. A circuit-modeled version needs three steps:

1. Put the nonlinear elements where the circuit has them.
2. Solve the resulting implicit equation per sample.
3. Handle aliasing.

I read your `.h` and `.cpp`. The notes below come from my own knowledge. I didn't search for papers or repos, so check the references and links yourself.

## 1. Where the nonlinearities go

In an analog SVF there are three typical places:

- **At each integrator input.** An OTA-based SVF, such as the Oberheim SEM, many OTA-C designs, or the CEM/SSM chips, saturates the current into the capacitor. The cutoff is `g`, and the integrator becomes `dv/dt = ω·tanh(input)`. This is the most characteristic and most important place.
- **In the damping/resonance path** (the `r*yB` term). This limits the resonance and makes the filter "bloom" instead of blowing up. It is the standard place in Zavalishin's nonlinear SVF.
- **At the input and output stages**, for drive, op-amp clipping, and so on. These are cheap, but they are static waveshapers and don't interact with the filter state.

Your TDF2 trapezoidal form maps onto this directly. With `u = yH`, a nonlinear version of your `getPartialOutputs` is:

```
yB = z1 + g*tanh(u)
yL = z2 + g*tanh(yB)
u  = in - r*yB - yL        // implicit: u appears on both sides
```

Your linear code solves this in closed form: `u = (in - c*z1 - z2)*s`. With the tanh in place, you have to solve a scalar nonlinear equation per sample.

The state update stays the same. `z1 = 2*yB - z1` is still equal to `z1 + 2*g*tanh(u)`, so the trapezoidal rule remains intact.

## 2. Solving the implicit equation

Define `F(u) = u - (in - r*yB(u) - yL(u))`. Then:

```
dyB/du = g * t'(u)
dyL/du = g * t'(yB) * dyB/du
F'(u)  = 1 + r*dyB/du + dyL/du
```

Here `t'(x) = 1 - tanh²(x)`. Since `t' ≥ 0` and `r, g > 0`, you get `F' ≥ 1`. The problem is therefore well conditioned, and Newton converges reliably.

```cpp
// Sketch only. Assumes TSig is a float type and tanh is available.
inline void getPartialOutputsNL(TSig in, TSig* yL, TSig* yB, TSig* yH)
{
  TSig u = (in - c*z1 - z2) * s;           // warm start: the linear solution
  for(int i = 0; i < numIterations; i++)   // typically 1..3 are enough
  {
    TSig tu  = tanh(u);
    TSig b   = z1 + g*tu;
    TSig tb  = tanh(b);
    TSig l   = z2 + g*tb;
    TSig F   = u - (in - r*b - l);
    TSig db  = g*(1 - tu*tu);
    TSig dl  = g*(1 - tb*tb)*db;
    u -= F / (1 + r*db + dl);
  }
  TSig tu = tanh(u);
  *yH = u;
  *yB = z1 + g*tu;
  *yL = z2 + g*tanh(*yB);
  z1 = 2 * *yB - z1;
  z2 = 2 * *yL - z2;
}
```

Alternatives, from cheapest to most accurate:

- **Linearize once per sample.** Replace `tanh(x)` by `x * tanh(x0)/x0` around a predicted operating point `x0`. This reduces to your existing closed-form solution with per-sample modified `g` values. It is cheap but slightly inexact.
- **Predict, then solve.** Predict the state or the nonlinearity input from the previous samples. Teemu Voipio and others posted this approach on the KVR DSP forum. It costs one sample of delay inside the nonlinearity, and it works surprisingly well when oversampled.
- **Newton with 1–3 iterations from a warm start.** The sketch above does this. A good initial guess matters most, and you can use the previous `u` or the linear solution.
- **Rational or polynomial saturators** instead of `tanh`. They are cheaper, and the derivative is simple. Examples are `x/(1+|x|)` and `x/sqrt(1+x²)`. The OTA tanh shape is nice, but it is not sacred.

## 3. Things that matter for the sound

- **Signal scaling.** The nonlinearity only does anything relative to its natural voltage scale. Add an input gain or drive parameter and decide what "0 dBFS" means in circuit volts.
- **Cutoff shift with level.** Saturating OTAs lower the effective integrator gain at high levels. This is a big part of the "analog" character, and it falls out of the model automatically.
- **Resonance behavior.** Compare `r*tanh(yB)` against the plain `r*yB` linear damping. The saturated version gives self-oscillation that settles at a bounded amplitude.
- **Aliasing.** Use 2x–4x oversampling, or antiderivative antialiasing (ADAA). ADAA is trickier for stateful systems, but Holters has a paper on exactly that. Since the nonlinearity is inside a feedback loop, oversampling is the more robust choice.
- **Modulation.** Check the DC response when switching the cutoff, as in your Wishnick note. Trapezoidal TDF2 states can pop when you modulate `g`. A state-variable form (storing the integrator output) behaves better. Your `z1 = 2*yB - z1` form already stores a "current-like" state, which is good.
- **Class design.** Keep the bell and shelf modes in the linear class. They are not analog circuits. For the nonlinear class, use LP/BP/HP/notch, which come from the mix coefficients `aL, aB, aH`. You can put the mix after the nonlinearities, as you do now.

## 4. Literature

I'm confident these exist. Please verify the exact titles and venues.

- **Vadim Zavalishin, *The Art of VA Filter Design*** (free from Native Instruments). It has chapters on nonlinear filters, including saturation in the SVF and Moog ladder, and on solving instantaneously nonlinear feedback loops.
- **Andrew Simper (Cytomic).** His SVF technical papers cover the linear trapezoidal SVF. They are a good baseline for comparing your implementation.
- **Federico Fontana and Marco Civolani**, "Modeling of the EMS VCS3 voltage-controlled filter as a nonlinear filter network" (IEEE TASLP, 2010). Fontana also has earlier work on nonlinear filter networks with delay-free loops.
- **Antti Huovilainen**, "Non-linear digital implementation of the Moog ladder filter" (DAFx-04). It is classic but uses a unit delay in the loop.
- **Stefano D'Angelo and Vesa Välimäki**, "An improved virtual analog model of the Moog ladder filter" (ICASSP 2013), and the "Generalized Moog ladder filter" papers (IEEE TASLP, 2014).
- **David Yeh**, "Automated physical modeling of nonlinear audio circuits for real-time audio effects" (Parts I and II), which introduces the DK method.
- **Martin Holters and Udo Zölzer**, papers on generic nonlinear state-space modeling and on antiderivative antialiasing for stateful systems.
- **Kurt Werner and colleagues**, on wave digital filters with multiple nonlinearities (DAFx 2015). WDFs are an alternative to the state-space approach.
- **Parker, Esqueda and Bergner**, on antiderivative antialiasing for waveshaping (DAFx 2016).

## 5. Open-source references

These are worth reading. I haven't checked their current state.

- **Surge XT** (`sst-filters`). It has many nonlinear VA filters, including ladder, diode ladder, and OBXD-style SVFs.
- **ChowDSP** (Jatin Chowdhury). It has WDF libraries and many circuit-modeled plugins.
- **Faust libraries** (`vaeffects.lib`, `misceffects.lib`), with nonlinear VA filters contributed by Eric Tarr and others.
- **Mutable Instruments** (`stmlib`, Rings, Warps). The SVF there is linear, but it is clean code and has simple tricks for mixing in saturation.
- **Airwindows**, for lightweight saturator ideas.
- **KVR DSP forum.** Posts from Mystran, Teemu Voipio, and Urs Heckmann cover cheap ZDF with nonlinearities and the predictor approach.

## Suggested path

1. Create a separate class, for example `rsStateVariableFilterNL`, with the same setup interface for LP, HP, BP and notch.
2. First put `tanh` at the two integrator inputs and solve with Newton. Check the linear limit: for small signals, the output must match your existing class exactly.
3. Add saturation in the damping path, and then input and output drive.
4. Add 2x oversampling and compare the result against the unoversampled one.
5. Optionally replace `tanh` with a cheaper rational approximation once you like the sound.

I can also prepare a PR with the nonlinear class and a unit test that checks the small-signal limit against `rsStateVariableFilter`. Tell me if you want that.

## User

Great! Thank you! I will test this code myself. This is the version where the nonlinearity sits at each integrator input, right? Can you give me also the versions where it sits in the other common places? And could you add comments to the code that clearly tell me, what quantities the variables represent?

## Assistant

Yes, the earlier code is the version with the nonlinearity at each integrator input. Below are that version again and the other placements, all with the same naming and comments. I haven't compiled or run any of this, so test the small-signal limit against your linear class first.

## Notation used in all variants

```
in   : input sample x[n]
u    : highpass output yH, also the signal that drives the first integrator
yB   : bandpass output (output of integrator 1)
yL   : lowpass output  (output of integrator 2)
z1   : state of integrator 1 (trapezoidal "memory", same as in your linear class)
z2   : state of integrator 2
g    : integrator gain tan(w/2)
r    : damping = 1/Q (your class calls this r; c = g + r)
t(x) : saturation function, here tanh(x)
t'(x): its derivative, here 1 - tanh(x)^2
```

The feedback loop in your linear filter is `u = in - r*yB - yL`. Each variant below puts a `t` somewhere in this loop and solves for `u` by Newton iteration.

## Shared pieces

```cpp
// Members to add to the class:
int  numIterations = 2;   // Number of Newton iterations (1..3 is usually enough)
TSig drive         = 1;   // Input gain: how hard the signal hits the nonlinearities

// Derivative of tanh, given the already computed value th = tanh(x):
static TSig tanhDerivative(TSig th) { return 1 - th*th; }
```

## Variant A: Saturation at the integrator inputs (OTA-style)

This models the OTA output current saturating, as in the SEM and OTA-C filters. It also makes the cutoff shift with level.

```cpp
inline void getPartialOutputsNL_A(TSig in, TSig* yL, TSig* yB, TSig* yH)
{
  TSig x = drive * in;                      // drive-scaled input

  // Initial guess for u: the solution of the linear filter (closed form)
  TSig u = (x - c*z1 - z2) * s;

  // Solve F(u) = u - (x - r*yB(u) - yL(u)) = 0 by Newton iteration:
  for(int i = 0; i < numIterations; i++)
  {
    TSig tu  = tanh(u);                     // saturated input of integrator 1
    TSig b   = z1 + g*tu;                   // yB as function of u
    TSig tb  = tanh(b);                     // saturated input of integrator 2
    TSig l   = z2 + g*tb;                   // yL as function of u

    TSig F   = u - (x - r*b - l);           // residual of the loop equation
    TSig db  = g * tanhDerivative(tu);      // d yB / d u
    TSig dl  = g * tanhDerivative(tb) * db; // d yL / d u
    TSig dF  = 1 + r*db + dl;               // dF/du, always >= 1 (well conditioned)

    u -= F / dF;                            // Newton step
  }

  // Final outputs from the converged u:
  TSig tu = tanh(u);
  *yH = u;
  *yB = z1 + g*tu;
  *yL = z2 + g*tanh(*yB);

  // Trapezoidal state update (same as the linear filter):
  z1 = 2 * *yB - z1;                        // == z1 + 2*g*tanh(yH)
  z2 = 2 * *yL - z2;                        // == z2 + 2*g*tanh(yB)
}
```

## Variant B: Saturation in the damping/resonance path

The integrators stay linear. Only the feedback `r*yB` is saturated, so the resonance is bounded and self-oscillation settles at a finite amplitude. The equation is scalar and cheap.

```cpp
inline void getPartialOutputsNL_B(TSig in, TSig* yL, TSig* yB, TSig* yH)
{
  TSig x = drive * in;

  // Initial guess: the linear solution
  TSig u = (x - c*z1 - z2) * s;

  // The integrators are linear here, so as functions of u:
  //   yB(u) = z1 + g*u
  //   yL(u) = z2 + g*yB(u) = z2 + g*z1 + g*g*u
  // Loop equation: F(u) = u - (x - r*tanh(yB(u)) - yL(u)) = 0
  for(int i = 0; i < numIterations; i++)
  {
    TSig b   = z1 + g*u;                    // yB as function of u
    TSig tb  = tanh(b);                     // saturated damping feedback signal
    TSig l   = z2 + g*b;                    // yL as function of u

    TSig F   = u - (x - r*tb - l);          // residual
    TSig dF  = 1 + r*g*tanhDerivative(tb) + g*g;  // dF/du, always >= 1 + g*g

    u -= F / dF;
  }

  *yH = u;
  *yB = z1 + g * *yH;
  *yL = z2 + g * *yB;

  z1 = 2 * *yB - z1;
  z2 = 2 * *yL - z2;
}
```

If you want a more circuit-like version, saturate the whole feedback sum instead, for example `u = x - tanh(r*yB + yL)`. This models an amplifier stage limiting the summing node. The derivative changes accordingly.

## Variant C: Static saturation at input and/or output

There is no feedback interaction here. These are waveshapers around the linear core. They are cheap, but they don't shift the cutoff or shape the resonance. They are useful for drive and for modeling op-amp stages.

```cpp
inline TSig getSampleNL_C(TSig in)
{
  TSig x = tanh(drive * in);                // input stage saturation
  TSig yL, yB, yH;
  getPartialOutputs(x, &yL, &yB, &yH);      // your existing linear core
  TSig y = aH*yH + aB*yB + aL*yL;           // mix as in getSample()
  return tanh(y);                           // output stage saturation (optional)
}
```

An alternative is to saturate each of `yL`, `yB`, `yH` individually before mixing. That models the output buffers of a multimode circuit. The internal states are unaffected, so the stability properties of the linear filter stay the same.

## Variant D: A + B combined

This is the most realistic of the four. It has OTA saturation in both integrators and a saturated damping path.

```cpp
inline void getPartialOutputsNL_D(TSig in, TSig* yL, TSig* yB, TSig* yH)
{
  TSig x = drive * in;
  TSig u = (x - c*z1 - z2) * s;             // initial guess: linear solution

  // Loop equation: F(u) = u - (x - r*tanh(yB(u)) - yL(u)) = 0 with
  //   yB(u) = z1 + g*tanh(u)
  //   yL(u) = z2 + g*tanh(yB(u))
  for(int i = 0; i < numIterations; i++)
  {
    TSig tu  = tanh(u);                     // saturated input of integrator 1
    TSig b   = z1 + g*tu;                   // yB(u)
    TSig tb  = tanh(b);                     // used both for integrator 2 and damping
    TSig l   = z2 + g*tb;                   // yL(u)

    TSig F   = u - (x - r*tb - l);          // residual
    TSig db  = g * tanhDerivative(tu);      // d yB / d u
    TSig dtb = tanhDerivative(tb) * db;     // d tanh(yB) / d u
    TSig dF  = 1 + r*dtb + g*dtb;           // dF/du (the damping and integrator 2
                                            // terms share the same derivative here)
    u -= F / dF;
  }

  TSig tu = tanh(u);
  *yH = u;
  *yB = z1 + g*tu;
  *yL = z2 + g*tanh(*yB);

  z1 = 2 * *yB - z1;
  z2 = 2 * *yL - z2;
}
```

Here the damping term and integrator 2 use the same `tanh(yB)`. That is a simplification, and a real circuit may use different saturation scales for them. If you want them separate, give each its own scale factor, for example `tanh(yB * k)`, and adjust the derivative.

## Notes for testing

- **Small-signal check.** With a tiny `drive`, scale the output back up and compare against `getPartialOutputs()`. The results should agree, and Newton should converge in one iteration.
- **Iteration count.** Compare 1, 2 and 8 iterations on a loud resonant signal. If 2 is audibly equal to 8, use 2.
- **Saturation scale.** Real circuits saturate at about 25 mV of differential input for an OTA. In normalized units, `drive` plays this role.
- **Stability.** Because `F' ≥ 1`, Newton should not diverge. If you see occasional spikes at extreme settings, clamp the step or fall back to the linear solution.
- **Aliasing.** Try 2x and 4x oversampling around the whole loop.

Variant D with `drive` and a mild output saturator is where I'd start for an analog-style sound. Tell me if you'd like a unit test or a PR.