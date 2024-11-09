// Origin:
// https://www.u-he.com/downloads/UrsBlog/fourPoleInv_Cascade_Newton_Vectorial.h

/*
	(c) 2016 by Urs Heckmann www.u-he.com
	
	License: I hereby place this code into the Public Domain - use and modify as you wish

	This class demonstrates how to use Newton's method to iteratively optimize all unknowns at once! This is a bit more involved mathematically but it's also much faster in extreme settings (high resonance, high filter cutoff).
	
	Because we do everything in one single loop there is no advantage in having a dedicated class for single pole filters.

*/

#ifndef uh_max
#define uh_max( A, B ) ( (A) > (B) ? (A) : (B) )
#endif

class fourPoleInv_Cascade_Newton_Vectorial
{
public:
	
	float s1, s2, s3, s4;		// filter state
	float v1, v2, v3, v4;		// pole voltages (used as initial values during iteration)
	
	float g;					// filter cutoff coefficient
	float k4Pole;				// filter resonance coefficient
		
	void clear()
	{
		s1 = s2 = s3 = s4 = 0.f;
		v1 = v2 = v3 = v4 = 0.f;
		setParams( 1000.f, 0.f, 44100.f );
	}
	
	void setParams( float cutoffHz, float resonance, float sampleRate )
	{	
		g			= tan( PI * cutoffHz / sampleRate );		
		k4Pole		= -resonance * 4.5f; // twice the resonance
	}
	
	float tick( float inSample )
	{		
		// init with values form previous sample
		
		float v1E = v1;
		float v2E = v2;
		float v3E = v3;
		float v4E = v4;
		
		// iterate until we find a good convergence 
		
		float error = 1.f; // at least one run
		
		int numiterations = 0;
		
		while ( error > (1.f/65535.f) ) // good enough?
		{			
			// Step 1: Calculate the tanh terms and store in variables for later use
			
			float tv1E = tanh( -v1E - inSample - k4Pole * v4E );
			float tv2E = tanh( -v2E - v1E );
			float tv3E = tanh( -v3E - v2E ) ;
			float tv4E = tanh( -v4E - v3E );
			
			// Step 2: run each stage, result is xn = Vn - VnE, i.e. the vectorial error function
			
			float x1 = g * ( tv1E ) + s1 - v1E;
			float x2 = g * ( tv2E ) + s2 - v2E;
			float x3 = g * ( tv3E ) + s3 - v3E;
			float x4 = g * ( tv4E ) + s4 - v4E;
			
			/*	Step 3: Compute terms in Jacobian Matrix
			 
				First we put the 4 equations for x1...x4 into our CAS, like this:
			 
				g * tanh( -Vin - k4pole * v4E - v1E ) + s1 - v1E
				g * tanh( v1E - v2E ) + s2 - v2E
				g * tanh( v2E - v3E ) + s3 - v3E
				g * tanh( v3E - v4E ) + s4 - v4E
			 
				Then we calculate the Jacobian Matrix, i.e. a Matrix that contains the derivative of our vectorial error function:
			 
				|	-g*(-1+tv1E^2)-1, 		0, 						0, 						-g*k4Pole*(1-tv1E^2)	|
				|	-g*(1-tv2E^2), 			-g*(-1+tv2E^2)-1, 		0, 						0						|
				|	0, 						-g*(1-tv3E^2),			-g*(-1+tv3E^2)-1, 		0						|
				|	0, 						0, 						-g*(1-tv4E^2), 			-g*(-1+tv4E^2)-1		|
			 
				For ease of use, we substitute the cells with variables (and discart the 0 entries in the following on computation)
			 
				|	m11,	0,		0,		m14	|
				|	m21,	m22,	0, 		0	|
				|	0,		m32,	m33,	0	|
				|	0,		0,		m43,	m44	|
			 */
			
			float m11 = -g*(1.f-tv1E*tv1E)-1.f;
			float m14 = -g*(1.f-tv1E*tv1E)*k4Pole;
			float m21 = -g*(1.f-tv2E*tv2E);
			float m22 = -g*(1.f-tv2E*tv2E)-1.f;
			float m32 = -g*(1.f-tv3E*tv3E);
			float m33 = -g*(1.f-tv3E*tv3E)-1.f;
			float m43 = -g*(1.f-tv4E*tv4E);
			float m44 = -g*(1.f-tv4E*tv4E)-1.f;
			
			/*	Step 4: Solve linear system
			 
				Newton Raphson is x[n+1] = x[n] + f(x[n])/f'(x[n]) 
				
				This works with vectors as well, where f'(x[n]) is the Jacobian Matrix. Our x is of course vE, our f(x) is x
				
				However, to divide by  a Matrix we need to invert it. Instead of inverting, we change the terms around a bit to bring the Matrix on top:
			 
				JF(vE)y = -F(vE)
				vE[ n + 1 ] = vE[ n ] + y
				
				As a result we get a system of linear equations:
			 
					y1*m11 + y4*m14 = -x1
					y1*m21 + y2*m22 = -x2
					y2*m32 + y3*m33 = -x3
					y3*m43 + y4*m44 = -x4
				
				which, after some reordering and substituting we can solve for y1, y2, y3, y4
				
				(this is the point where you'll finally be thankful for a computer algebra software) 
			 */
			
			float y1 =  (m14*m22*m33*x4-m14*m22*m43*x3+m14*m32*m43*x2-m22*m33*m44*x1)/(m11*m22*m33*m44-m14*m21*m32*m43);
			float y2 = -(m11*m33*m44*x2+m14*m21*m33*x4-m14*m21*m43*x3-m21*m33*m44*x1)/(m11*m22*m33*m44-m14*m21*m32*m43);
			float y3 = -(m11*m22*m44*x3-m11*m32*m44*x2-m14*m21*m32*x4+m21*m32*m44*x1)/(m11*m22*m33*m44-m14*m21*m32*m43);
			float y4 = -(m11*m22*m33*x4-m11*m22*m43*x3+m11*m32*m43*x2-m21*m32*m43*x1)/(m11*m22*m33*m44-m14*m21*m32*m43);
			
			// Step 5: Improve our guess. Add y1-y4 to v1E to v4E as new estimates
			
			v1E += y1;
			v2E += y2;
			v3E += y3;
			v4E += y4;
			
			// Step 6: See how good we are, stop if good enough (can be done on either x or y, here it's on y)
			
			float error1 = uh_max( fabs( y1 ), fabs( y2 ) );
			float error2 = uh_max( fabs( y3 ), fabs( y4 ) );
			error = uh_max( error1, error2 );
			
			if( ++numiterations > 20 )
			{
				// might assert here and see what's wrong
				
				break;
			}
		}
		
		// store vs as starting point for next sample
		
		v1 = v1E;
		v2 = v2E;
		v3 = v3E;
		v4 = v4E;
		
		// update state using trapezoidal rule
		
		s1 = 2 * v1 - s1;
		s2 = 2 * v2 - s2;
		s3 = 2 * v3 - s3;
		s4 = 2 * v4 - s4;
		
		return v4;
	}
};
