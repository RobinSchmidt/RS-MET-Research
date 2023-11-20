
// Simple implementation of band-limited steps with adjustable high-pass and
// low-pass filtering, for efficient band-limited synthesis. Refer to the web
// site for more:
//
//     http://www.slack.net/~ant/bl-synth
//
// Shay Green <hotpop.com@blargg> (swap to e-mail)

#include "Wave_Writer.h"
#include <math.h>

double const low_pass  = 0.999; // lower values filter more high frequency
double const high_pass = 0.990; // lower values filter more low frequency

int const phase_count = 32; // number of phase offsets to sample band-limited step at
int const step_width  = 16; // number of samples in each final band-limited step

static float steps [phase_count] [step_width]; // would use short for speed in a real program

void init_steps()
{
	// Generate master band-limited step by adding sine components of a square wave
	int const master_size = step_width * phase_count;
	float master [master_size]; // large; might want to malloc() instead
	for ( int i = 0; i < master_size; i++ )
		master [i] = 0.5;
	
	double gain = 0.5 / 0.777; // adjust normal square wave's amplitude of ~0.777 to 0.5
	int const sine_size = 256 * phase_count + 2;
	int const max_harmonic = sine_size / 2 / phase_count;
	for ( int h = 1; h <= max_harmonic; h = h + 2 )
	{
		double amplitude = gain / h;
		double to_angle = 3.14159265358979323846 * 2 / sine_size * h;
		for ( int i = 0; i < master_size; i++ )
			master [i] += sin( (i - master_size / 2) * to_angle ) * amplitude;
		
		gain = gain * low_pass;
	}
	
	// Sample master step at several phases
	for ( int phase = 0; phase < phase_count; phase++ )
	{
		double error = 1.0;
		double prev = 0.0;
		for ( int i = 0; i < step_width; i++ )
		{
			double cur = master [i * phase_count + (phase_count - 1 - phase)];
			double delta = cur - prev;
			error = error - delta;
			prev = cur;
			steps [phase] [i] = delta;
		}
		
		// each delta should total 1.0
		steps [phase] [step_width / 2 - 1] += error * 0.5;
		steps [phase] [step_width / 2    ] += error * 0.5;
	}
}

long const sample_rate = 44100;
long const buf_size = sample_rate * 4;
static float buf [buf_size + step_width];

void add_step( double time, double delta )
{
	int whole = floor( time );
	int phase = (time - whole) * phase_count;
	for ( int i = 0; i < step_width; i++ )
		buf [whole + i] += steps [phase] [i] * delta;
}

int main()
{
	init_steps();
	
	// Add square wave of slowly lowering frequency
	double period = 1;
	double delta = 0.5;
	for ( double time = 0; time < buf_size; time = time + period )
	{
		add_step( time, delta );
		period = period * 1.0005; // slowly lower frequency
		delta = -delta;
	}
	
	// Sum output buffer
	double sum = 0;
	for ( long i = 0; i < buf_size; i++ )
	{
		sum = sum + buf [i];
		buf [i] = sum;
		sum = sum * high_pass;
	}
	
	// Write to wave file
	Wave_Writer wave( sample_rate );
	wave.write( buf, buf_size );
	
	return 0;
}

