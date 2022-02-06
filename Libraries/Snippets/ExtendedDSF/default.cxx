// =======================================================
// Name: Extended DSF Algorithms by Walter H. Hackett
// =======================================================
// Description: A set of integrated algorithms by 
// through the use of Discrete Summation Formulaes/
// This is acheived by finding the closed-form of 
// harmonic summations witrh various augmentations to
// both amplitude and phase.
// 
// Through this concept it is possible to create many
// bandlimited waveforms directly or via compositing
// multiple Extended DSF algorithms that can start and end
// at arbitrary offsets to the fundamental.
//
// Autofiltered waveforms using exponential and linear 
// functions and more complex waveforms and animatable
// effects such as those seen in major additive synthesis
// applications, with all the benefits of closed-form 
// summations (Extended DSF).
//
// Here is my first batch of oscillators based on my
// research so far.
//
// - Walter H. Hackett
// 
// Appreciation for inspiration and assistance goes to
// my family, Jess of Plugins4Free, 
// Elan Hickler of SoundEmote,
// James A. Moorer, Julius O Smith, Tim Stilson, 
// Burkhard Reike, Bastiaan Barth of SolidTrax, 
// Steve Duda (and Joel Zimmerman) of Xfer Records, 
// Alexandre Bique of BitWig/CLAP,
// Andy Simper of Cytomic, Urs Heckmann of U-he,
// Teemu Voipio, Jacob of Cableguys, Markus of Tone2, 
// Jeff McClintock of Synthedit/GMPI, Siraj Raval,
// Spogg of Flowstoners, Blue Cat Audio, fukuroda, 
// and musicdsp archive.
// =======================================================

                           
                                

string name="Extended DSF Oscillators";
string description=name;
blit_saw_oscillator _blit_saw;

array<double> inputParameters(2,0);
array<string> inputParametersUnits={, ""};
array<string> inputParametersNames={"Modulate", "Mode"};
array<int>    inputParametersSteps={,15};
array<double> inputParametersMax={,14.001};
array<string>  inputParametersEnums={,
    "Autofilter Engineer's Saw;"+
    "Autofilter Musician's Saw;"+
    "Autofilter Engineer's Square;"+
    "Autofilter Musician's Square;"+
    "Autofilter Engineer's Triharmonic;"+
    "Autofilter Musician's Triharmonic;"+
    "Autofilter Walter's Saw;"+
    "Phase Rotation Saw;"+
    "Pure Engineer's Saw;"+
    "Pure Musician's Saw;"+
    "Pure Engineer's Square;"+
    "Pure Musician's Square;"+
    "Walter's Saw;"+
    "Linear Engineer's Saw;"+
    "Linear Engineer's Square"
    };

int mode = 0;
double k = 1.0;
const double tau = 3.14159265358979 * 2.0;

class blit_saw_oscillator_note
{
    double t; // current time
    double value; // current value
    int n; // nyquist limit
    double dt; // delta t
    int note_no; // note number [0,127]
    double velocity; // velocity [0,1]
    
    blit_saw_oscillator_note()
    {
        t = 0.0;
        value = 0.0;
        n = 0;
        dt = 0.0;
        note_no = 0;
        velocity = 0.0;
    }
};

class blit_saw_oscillator
{
    array<blit_saw_oscillator_note> _notes(16);
    uint _active_note_count;
    double _pitchbend;

    blit_saw_oscillator()
    {
        _active_note_count = 0;
        _pitchbend = 0.0;
    }

    void trigger(const MidiEvent& evt)
    {
        if( _active_note_count < _notes.length )
        {
            blit_saw_oscillator_note@ note = _notes[_active_note_count];

            note.note_no = MidiEventUtils::getNote(evt);
            note.velocity = MidiEventUtils::getNoteVelocity(evt)/127.0;
            note.value = 0.0;
            note.t = 0.5;

            double freq = 440.0*(pow(2.0, (note.note_no + _pitchbend - 69.0) / 12.0));
            note.n = int(sampleRate / 2.0 / freq);
            note.dt = freq / sampleRate;

            ++_active_note_count;
        }
    }

    void update_pitchbend(const MidiEvent& evt)
    {
        _pitchbend = MidiEventUtils::getPitchWheelValue(evt)/4096.0;

        for (uint i = 0; i < _active_note_count; ++i)
        {
            blit_saw_oscillator_note@ note = _notes[i];

            double freq = 440.0*(pow(2.0, (note.note_no + _pitchbend - 69.0) / 12.0));
            note.n = int(sampleRate / 2.0 / freq);
            note.dt = freq / sampleRate;
        }
    }

    void release(const MidiEvent& evt)
    {
        int note_no = MidiEventUtils::getNote(evt);
        uint idx;
        for(idx = 0; idx < _active_note_count; ++idx)
        {
            if( _notes[idx].note_no == note_no && idx < _active_note_count )
            {
                if( idx < _active_note_count - 1) _notes[idx] = _notes[_active_note_count - 1];
                --_active_note_count;
            }
        }
    }

    bool is_silent()
    {
        return _active_note_count == 0;
    }

    // AutoFilter Engineer's Saw
    double autofilterSawEng(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m);
        return 2.0 * (-pow(k,n+1)*cos((n+1)*x)-k*(-pow(k,n+1)*cos(x*n)+k-cos(x)))/(1-2*cos(x)*k+k*k);
    } 

    // Autofilter Musician's Saw
    double autofilterSawMus(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m);
        return (sin(x)*k+pow(k,n+1)*(k*sin(x*n)-sin((n+1)*x)))/(1-2*cos(x)*k+k*k);
    } 

    // AutoFilter Engineer's Square
    double autofilterSquEng(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m * 0.5);
        return 4.0 * ((k*k*k*k-k*k)*cos(x)-pow(pow(k,n+1),2)*(cos(x*(2*n-1))*k*k-cos(x*(2*n+1))))/(-k*k*k*k*k+2*cos(2*x)*k*k*k-k);
    } //

    // Autofilter Musician's Square
    double autofilterSquMus(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m * 0.5);
        return 2.0*((-k*k*k*k-k*k)*sin(x)-pow(pow(k,n+1),2)*(sin(x*(2*n-1))*k*k-sin(x*(2*n+1))))/(-k*k*k*k*k+2*cos(2*x)*k*k*k-k);
    } //

    // AutoFilter Engineer's Triharmonic
    double autofilterTriEng(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m / 3.0);
        return 3.0 * (cos(x*(3*n-2))*pow(pow(k,n+1),3)*k*k*k-k*k*k*k*k*k*cos(2*x)+cos(x)*k*k*k-cos(x*(3*n+1))*pow(pow(k,n+1),3))/k*k/(1+k*k*k*k*k*k-2*cos(3*x)*k*k*k);
    } //

    // AutoFilter Musician's Triharmonic
    double autofilterTriMus(double t, int m)    
    {
        double x = t * tau;
        double n = floor(m / 3.0);
        double kcube = pow(k,3);
        return 3.0*(pow(pow(k,n+1),3)*sin(x*(3*n-2))*kcube-pow(pow(k,n+1),3)*sin(x*(3*n+1))+kcube*(kcube*sin(2*x)+sin(x)))/k*k/(1+pow(k,6)-2*cos(3*x)*kcube);
    } //

    // Autofilter Walter's Saw
    double autofilterWalterSaw(double t, int m)    
    {
        double x = t * tau;
        double n = floor((m-1) * 0.5);

        return -2*(2*k*sin(x)*cos(x)*pow(pow(k,n+1),2)*(-2*k*sin(x)+k*k-1)*pow(cos(x*(n+1)),2)-2*pow(pow(k,n+1),2)
            *((-2*k*k*sin(x)+pow(k,3)+k)*pow(cos(x),2)-(1+k*k)*(k-sin(x)))*sin(x*(n+1))*cos(x*(n+1))-k*sin(x)
            *cos(x)*(pow(pow(k,n+1),2)+1)*(-2*k*sin(x)+k*k-1))/(-4*k*k*sin(x)*pow(cos(x),2)+pow(1+k*k,2)*sin(x));
    } //

    // Phase Rotatation Saw
    double phaseRotatingSaw(double t, int m) 
    {
        double x = t * tau;
        double n = floor(m);
        return 1.414 * (((sin(n*x+k+x)-sin(n*x+k)+sin(k-x)-sin(k))/(-2+2*cos(x))) - sin(k));
    } // 

    // Pure Harmonic Engineer's Saw
    double pureSawEng(double t, int n)
    {
        return 2.0*(((sin(PI*4*t*(floor(n)*2+1)/4))/sin(t/4*4*PI)-1)/2);
    }

    // Pure Harmonic Musician's Saw
    double pureSawMus(double t, int n)
    {
        return (((cos(PI*t)-cos(t*(4*floor(n)+2)*PI/2))/sin(t*PI)) + 2) * 0.5 - 1.0;
    }

    // Pure Harmonic Engineer's Square
    double pureSquEng(double t, int n)
    {
        return 2.0*(sin(4*PI*t*floor(n*0.5))/sin(2*PI*t));
    }

    // Pure Harmonic Musician's Square
    double pureSquMus(double t, int n)
    {
        return (((1-cos(t*(floor(n/2)*16)*PI*0.25))/sin(t*PI*2)));
    }

    // Walter's Saw
    double pureWaltersSaw(double t, int m)
    {
        double x = t * tau;
        double n = floor((m-1) * 0.5);
        return -2.0 * ((-cos(x)*pow(cos(x*(n+1)),2)-sin(x*(n+1))*(sin(x)-1)*cos(x*(n+1))-((n+1)*pow(cos(x),2)+
            (n+1)*pow(sin(x),2)-n-2)*cos(x))/sin(x));
    }    

    // Linear Harmonic Engineer's Saw
    double f1(double x, double t)
    {
        double t1 = floor(t*2.0+2.0);
        if (int(floor(t)) % 2 == 0) return ( (1.0 + cos(x*t1*PI)) / cos(x*PI)) / t1;
        return ( (1.0 - cos(x*t1*PI)) / cos(x*PI)) / t1;
    }
    double linSawEng(double t, int n)
    {
        return (f1(t,n) / cos(t*PI)) - 1.0 ;
    }


    // Linear Harmonic Engineer's Square
    double f2(double x, double t)
    {
        double t1 = t*2.0-2.0;
        if (int(floor(t)) % 2 == 0) return ((1+cos(x*t1*PI))/cos(x*PI))/t1;
        return ((1-cos(x*t1*PI))/cos(x*PI))/t1;
    }

    double linSquEng(double t, int n)
    {
        return 2.0*(f2(t*2,floor(n*0.5+1.0)))/cos(t*PI*2) * sin(PI*t*2);
    }

    double process_sample()
    {
        double value = 0.0;
        for (uint i = 0; i < _active_note_count; ++i)
        {
            blit_saw_oscillator_note@ note = _notes[i];
            value += note.value * note.velocity;
            note.t += note.dt * 0.9999;
            note.t = note.t - floor(note.t);
            switch(mode)
            {
                case 0:
                    note.value = note.value*0.999 + (autofilterSawEng(note.t, note.n) )*note.dt;
                    break;
                case 1:
                    note.value = note.value*0.999 + (autofilterSawMus(note.t, note.n) )*note.dt;
                    break;
                case 2:
                    note.value = note.value*0.999 + (autofilterSquEng(note.t, note.n) )*note.dt;
                    break;
                case 3:
                    note.value = note.value*0.999 + (autofilterSquMus(note.t, note.n) )*note.dt;
                    break;
                case 4:
                    note.value = note.value*0.999 + (autofilterTriEng(note.t, note.n) )*note.dt;
                    break;
                case 5:
                    note.value = note.value*0.999 + (autofilterTriMus(note.t, note.n) )*note.dt;
                    break;
                case 6:
                    note.value = note.value*0.999 + (autofilterWalterSaw(note.t, note.n) )*note.dt;
                    break;
                case 7:
                    note.value = note.value*0.999 + (phaseRotatingSaw(note.t, note.n) )*note.dt;
                    break;
                case 8:
                    note.value = note.value*0.999 + (pureSawEng(note.t, note.n) )*note.dt;
                    break;
                case 9:
                    note.value = note.value*0.999 + (pureSawMus(note.t, note.n) )*note.dt;
                    break;
                case 10:
                    note.value = note.value*0.999 + (pureSquEng(note.t, note.n) )*note.dt;
                    break;
                case 11:
                    note.value = note.value*0.999 + (pureSquMus(note.t, note.n) )*note.dt;
                    break;
                case 12:
                    note.value = note.value*0.999 + (pureWaltersSaw(note.t, note.n) )*note.dt;
                    break;
                case 13:
                    note.value = note.value*0.999 + (linSawEng(note.t, note.n) )*note.dt;
                    break;
                case 14:
                    note.value = note.value*0.999 + (linSquEng(note.t, note.n) )*note.dt;
                    break;
            }
            
        }
        return value;
    }
};

void processBlock(BlockData& data)
{
    const MidiEvent@ event;
    if( 0 < data.inputMidiEvents.length )
    {
        @event = data.inputMidiEvents[0];
    }

    uint event_idx = 0;
    for(uint i=0; i<data.samplesToProcess; ++i)
    {
        while( @event != null && event.timeStamp <= double(i) )
        {
            MidiEventType evt_type = MidiEventUtils::getType(event);
            if( evt_type == kMidiNoteOn )
            {
                _blit_saw.trigger(event);
            }
            else if( evt_type == kMidiNoteOff )
            {
                _blit_saw.release(event);
            }
            else if( evt_type == kMidiPitchWheel )
            {
                _blit_saw.update_pitchbend(event);
            }

            ++event_idx;
            if( event_idx < data.inputMidiEvents.length )
            {
                @event = data.inputMidiEvents[event_idx];
            }
            else
            {
                @event = null;
            }
        }

        if( _blit_saw.is_silent() )continue;

        double value = _blit_saw.process_sample();
        for(uint ch = 0; ch < audioOutputsCount; ++ch)
        {
            data.samples[ch][i] = value;
        }
    }
}

void updateInputParameters()
{
    mode = int(inputParameters[1]);
    switch(mode)
    {
        case 0:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.35),14)*10000.0+1.0))+0.0000001;
            break;
        case 1:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.35),14)*10000.0+1.0))+0.0000001;
            break;
        case 2:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.35),14)*10000.0+1.0))+0.0000001;
            break;
        case 3:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.3),14)*10000.0+1.0))+0.0000001;
            break;
        case 4:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.43),14)*10000.0+1.0))+0.0000001;
            break;
        case 5:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.4),14)*10000.0+1.0))+0.0000001;
            break;
        case 6:
            k = 1.0-(1.0/(pow(((inputParameters[0])/2+0.3),14)*10000.0+1.0))+0.0000001;
            break;
        case 7:
            k = inputParameters[0] * tau;
            break;
        case 8:
            break;
        case 9:
            break;
        case 10:
            break;
        case 11:
            break;
        case 12:
            break;
        case 13:
            break;
        case 14:
            break;

    }
    
}

int getTailSize()
{
    return -1;
}


/** 
*  \file Midi.hxx
*  Midi utilities library for angelscript
* 
*  Created by Blue Cat Audio <services@bluecataudio.com>
*  Copyright 2011-2017 Blue Cat Audio. All rights reserved.
*
*/

/** Enumeration of Midi Events types currently supported by this library.
*
*/
enum MidiEventType
{
    kMidiNoteOff, ///< Note Off Event
    kMidiNoteOn, ///< Note On Event
    kMidiControlChange, ///< Control Change Event (CC)
    kMidiProgramChange, ///< Program Change Event
    kMidiPitchWheel,  ///< Pitch Wheel Event
    kMidiNoteAfterTouch,  ///< Note after touch
    kMidiChannelAfterTouch, ///< Channel after touch
    kUnknown ///< Other Events
};


/** Utility functions to handle Midi events.
*
*/
namespace MidiEventUtils
{
    /** Retrieves the type of the Midi event.
    *
    */
    MidiEventType getType(const MidiEvent& evt)
    {
        MidiEventType type=kUnknown;
        switch(evt.byte0 & 0xF0)
        {
        case 0x80:
            type=kMidiNoteOff;
            break;
        case 0x90:
            type=kMidiNoteOn;
            break;
        case 0xB0:
            type=kMidiControlChange;
            break;
        case 0xC0:
            type=kMidiProgramChange;
            break;
        case 0xE0:
            type=kMidiPitchWheel;
            break;
        case 0xA0:
            type=kMidiNoteAfterTouch;
            break;
        case 0xD0:
            type=kMidiChannelAfterTouch;
            break;
        }
        return type;
    }

    /** Set the type of the Midi event.
    *  @see MidiEventType enum for supported Midi events.
    */
    void setType(MidiEvent& evt,MidiEventType type)
    {
        switch(type)
        {
        case kMidiNoteOff:
            evt.byte0=0x80|(evt.byte0 & 0x0F);
            break;
        case kMidiNoteOn:
            evt.byte0=0x90|(evt.byte0 & 0x0F);
            break;
        case kMidiControlChange:
            evt.byte0=0xB0|(evt.byte0 & 0x0F);
            break;
        case kMidiProgramChange:
            evt.byte0=0xC0|(evt.byte0 & 0x0F);
            break;
        case kMidiPitchWheel:
            evt.byte0=0xE0|(evt.byte0 & 0x0F);
            break;
        case kMidiNoteAfterTouch:
            evt.byte0=0xA0|(evt.byte0 & 0x0F);
            break;
        case kMidiChannelAfterTouch:
            evt.byte0=0xD0|(evt.byte0 & 0x0F);
            break;
        }
    }

    /** Get the channel number of the event (1-16).
    *
    */
    uint8 getChannel(const MidiEvent& evt)
    {
        return (evt.byte0 & 0x0F)+1;
    }

    /** Set the channel number for the event (1-16).
    *
    */
    void setChannel(MidiEvent& evt, uint8 ch)
    {
        if(ch!=0)
            evt.byte0=(evt.byte0&0xF0)|((ch-1)&0x0F);
    }

    /** For a note event, retrieves the Midi note for the event (0-127).
    *
    */
    uint8 getNote(const MidiEvent& evt)
    {
        return evt.byte1&0x7F;
    }

    /** For a note event, sets the Midi note for the event (0-127).
    *
    */
    void setNote(MidiEvent& evt, uint8 note)
    {
        evt.byte1=(note&0x7F);
    }

    /** For a note event, retrieves the velocity for the event (0-127).
    *
    */
    uint8 getNoteVelocity(const MidiEvent& evt)
    {
        return evt.byte2 & 0x7F;
    }

    /** For a note event, sets the velocity for the event (0-127).
    *
    */
    void setNoteVelocity(MidiEvent& evt, uint8 velocity)
    {
        evt.byte2=velocity&0x7F;
    }

    /** For a CC (Control Change) event, gets the control number (0-127).
    *
    */
    uint8 getCCNumber(const MidiEvent& evt)
    {
        return evt.byte1 & 0x7F;
    }

    /** For a CC (Control Change) event, sets the control number (0-127).
    *
    */
    void setCCNumber(MidiEvent& evt,uint8 nb)
    {
        evt.byte1=nb&0x7F;
    }

    /** For a CC (Control Change) event, gets the control value (0-127).
    *
    */
    uint8 getCCValue(const MidiEvent& evt)
    {
        return evt.byte2 & 0x7F;	
    }

    /** For a CC (Control Change) event, sets the control value (0-127).
    *
    */
    void setCCValue(MidiEvent& evt,uint8 value)
    {
        evt.byte2=value & 0x7F;
    }

    /** For a Program Change event, gets the program number (0-127).
    *
    */
    uint8 getProgram(const MidiEvent& evt)
    {
        return evt.byte1&0x7F;
    }

    /** For a Program Change event, sets the program number (0-127).
    *
    */
    void setProgram(MidiEvent& evt, uint8 prog)
    {
        evt.byte1=prog&0x7F;
    }    

    /** For a pitch Wheel event, gets the pitch value (-8192 to +8192).
    *
    */
    int getPitchWheelValue(const MidiEvent& evt)
    {
        return (evt.byte2 & 0x7F)*128+(evt.byte1 & 0x7F)-64*128;	
    }

    /** For a pitch Wheel event, sets the pitch value (-8192 to +8192).
    *
    */
    void setPitchWheelValue(MidiEvent& evt, int value)
    {
        int midiValue=value+64*128;
        evt.byte1=midiValue&0x7F;
        evt.byte2=(midiValue/128)&0x7F;
    }

    /** For a channel after touch event, gets the after touch value (0-127)
    *
    */
    int  getChannelAfterTouchValue(const MidiEvent&  evt)
    {
        return evt.byte1&0x7F;
    }

    /** For a channel after touch event, sets the after touch value (0-127)
    *
    */
    void  setChannelAfterTouchValue(MidiEvent& evt, int value)
    {
        evt.byte1=(value&0x7F);;
    }
}
/** 
 *  \file Constants.hxx
 *  Common math constants for dsp scripting.
 * 
 *  Created by Blue Cat Audio <services@bluecataudio.com>
 *  Copyright 2014 Blue Cat Audio. All rights reserved.
 *
 */

/// Pi value
const double PI=3.141592653589793238462;
