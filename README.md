# Application recognizing main harmonic partials
This is source code of an application made to recognize pitches contained in polyphonic sounds.
Main tools used to achieve the result are fast Fourier transform and neural network using pytorch.
To achieve feasible computation time, many rules and properies of western music were used, for example twelve-tone equal temperament and its relation to harmonic partials.
## Required packages
To install required Python packages, use:
```console
  $ pip install -r requirements.txt 
```
To install fluidsynth package, use:
```console
  $ pacman -S fluidsynth
```
## Instructions
To display instructions, use the following command:
```console
  $ python main.py --help
```

