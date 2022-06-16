# Implementation of the cuFFT and FFTW libraries

This documentation is about implementing a GPU-accelerated FFT (Fast Fourier Transform) library in an OpenACC application interface, and also the FFTW library in a serial scenario. A Fortran code illustrating the implementation is provided. Details about the compilation process is described in the documentation.

The code was tested on both NVIDIA A100 and P100.

# References

- FFTW: https://www.fftw.org/

- cuFFT & cuFFTW: https://docs.nvidia.com/cuda/cufft/index.html

- OpenACC directives: https://www.nvidia.com/docs/IO/116711/OpenACC-API.pdf
