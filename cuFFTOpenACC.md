# GPU-accelerated library FFT

# Summary

In this documentation we provide an overview on how to implement the GPU-accelerated library FFT (Fast Fourier Transform) in OpenACC and OpenMP applications. Here we distinguish between two FFT libraries: cuFFT and cuFFTW. The cuFFT library is the NVIDIA-GPU based design, while cuFFTW is a porting version of the existing [FFTW](https://www.fftw.org/) library. In this tutorial, both libraries will be addressed with a special focus on the implementation of the cuFFT library. Specifically, the aim (focus) of this tutorial is to:
* Show how to incorporate the FFTW library in a serial code.
* Describe how to use the cuFFTW library.
* Show how to incorporate the cuFFT library in an OpenACC and OpenMP applications.
* Describe briefly how to enable cuFFT to run on OpenACC stream.
* Describe the compilation process of FFTW and cuFFT.

The implementation will be illustrated for a one-dimensional (1D) scenario and will be further described for 2D and 3D cases.


#### Table of Contents

- [Generality of FFT](#generality-of-fft)
- [Implementation of FFTW](#implementation-of-fftw)
- [Compilation process of FFTW](#compilation-process-of-fftw)
- [Implementation of cuFFT](#implementation-of-cufft)
- [Compilation process of cuFFT](#compilation-process-of-cufft)

# Generality of FFT

In general, the implementation of an FFT library is based on three major steps as defined below:

-Creating plans (initialization).
-Executing plans (create a configuration of a FFT plan having a specified dimension and data type).
-Destroying plans (to free the ressources associated with the FFT plans).

These steps necessitate specifying the direction, in which the FFT algorithm should be performed: forward or backward (or also inverse of FFT), and the dimension of the problem at hands as well as the precision (i.e. double or single precision); this is in addition to the nature of the data (real or complex) to be transformed.  

In the following, we consider a one-dimensional (1D) scenario, in which the execution is specified for a double precision complex-to-complex transform plan in the forward and backward directions. The Fortran code can be adjusted to run calculations of a single precision as well as of real-to-real/complex transform and can be further extended to multi-dimension cases (i.e. 2D and 3D). We first start with the FFT implementation in a CPU-serial scheme and further extend it to a GPU-accelerated case.  The implementation is illustrated for a simple example of a function defined in time-domain. Here we choose a sinus function (i.e. f(t)=sin(&omega;t) with &omega; is fixed at the value 2), and its FFT should result in a peak around the value &omega;=2 in the frequency domain.  

# Implementation of FFTW   

The implementation of the FFTW library is shown below and a detailed description of the library can be found here.

``` fortran
do while (max_err.gt.error.and.iter.le.max_iter)
   do j=2,ny-1
      do i=2,nx-1
          d2fx = f(i+1,j) + f(i-1,j)
          d2fy = f(i,j+1) + f(i,j-1)
          f_k(i,j) = 0.25*(d2fx + d2fy)
      enddo
   enddo
enddo   
```

As described above, one needs to initialize the FFT by creating plans, in which the.
Executing the plans requires specifying the transform direction: FFTWFORWARD for the forward direction or FFTWBACKWARD for the backward direction (inverse FFT). These direction parameters should be defined as an integer parameter. An alternative is to include the fftw.f file as a header (i.e. include …), which contains all parameters required for a general use of FFTW. In this case, the value of the direction parameter does not need to be defined.  

Note that when implementing the FFTW library, the data obtained from the backward direction need to be normalized by dividing the output array by the size of the data, while those of forward direction do not. This is only valid when using the FFTW library.

To check the outcome of the result in the forward direction, one can plot the function in the frequency-domain, which should display a peak around the value &w;=+2 and -2 as the function is initially symmetric. By performing the backward FFT of the obtained function, one should obtain the initial function displayed in time-domain (i.e. sin(2t)). This checking procedure should hold also when implementing a GPU version of the FFT library.

# Compilation process of FFTW

For the serial case, the FFTW library should be linked with fftw3 (i.e. `-lfftw3`) for the double precision, and fftw3f (i.e. `-lfftw3f`) for the single precision case. 

Load an intel module: e.g. 
To compile: 
```bash
ifort -lfftw3 -o fftw.serial fftw_serial.f90
```
To execute: 
```bash
./fftw.serial
```

# Implementation of cuFFT   

The cuFFTW library is part of the CUDA toolkit, and thus it is supported by the NVIDIA-GPU compiler. We consider the same scenario as described in the previous section. The cuFFT implementation is shown below:

``` fortran
do while (max_err.gt.error.and.iter.le.max_iter)
   do j=2,ny-1
      do i=2,nx-1
          d2fx = f(i+1,j) + f(i-1,j)
          d2fy = f(i,j+1) + f(i,j-1)
          f_k(i,j) = 0.25*(d2fx + d2fy)
      enddo
   enddo
enddo   
```

Similarly to the FFTW library, the implementation of the cuFFT library is based on creating plans, executing and destroying them. The difficulty however is how to combine the com host memory to the device memory without …

This is done in OpenACC by specifying the directive `host_data` together with the clause `use_device()`. Their use here enables overlapping the computation on the CPU-host and calling the cuFFT library routine that requires a GPU-device memory.

It requires including the header lines `use cufft` and `use openacc`.

Note that the cuFFT library uses CUDA streams for an asynchronous execution. On the other hand, OpenACC does not…It is therefore necessary to make cuFFT run on OpenACC streams. This is done by calling the routine `cufftSetStream()`, which is part of the cuFFT module. The routine includes the function `acc_get_cuda_stream()`, which enables identifying the CUDA stream.


The tables below summarize the [calling functions](https://docs.nvidia.com/hpc-sdk/compilers/fortran-cuda-interfaces/index.html) in  the case of a multi-dimension data having a simple or double complex data type.


Dimension| 1D | 2D | 3D |
-- | -- | -- | -- |
Creating a FFT plan | cufftPlan1D(plan,nx, FFTtype,1) | cufftPlan2d( plan, ny, nx, FFTtype) | cufftPlan3d( plan, nz, ny, nx, FFTtype ) |
-- | -- | -- | -- |

**Table 1.** *Creating FFT plans in 1D, 2D and 3D. Dimension executing a double precision complex-to-complex transform plan in the forward and backward directions. Nx is the size of a 1D array, nx and ny the size of a 2D array, and nx,ny,nz is the size of a 3D array. The FFTtype specifies the data type stored in the arrays in and out as described in the **Table 2**.*


Precision of the transformed plan | subroutine | FFTtype |
 -- | -- | -- |
Double precision complex-to-complex transform plan | cufftExecZ2Z( plan, in, out, direction ) | FFTtype=”CUFFT_Z2Z” |
-- | -- | -- |
Single precision complex-to-complex transform plan | cufftExecC2C( plan, in, out, direction ) | FFTtype=”CUFFT_C2C” |

**Table 2.** *The execution of a function using the cuFFT library. The direction specifies the FFT direction: “CUFFT_FORWARD” for forward FFT and “CUFFT_INVERSE” for backward FFT. The input data are stored in the array in, and the results of FFT for a specific direction are stored in the array out.*

Finally, the function `cufftDestroy(plan)` frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
 

# Compilation process of cuFFT

The cuFFT library is part of the CUDA toolkit. Therefore, the only modules are required to be load are NVHPC and CUDA modules.
 
Modules to be loaded:
On Betzy:
To compile: it requires linking the cuFFT library (`-lcufft`) and adding the CUDA version library to the syntax of the compilation (`-cudalib=cufft`)
We compile using the NVIDIA Fortran compiler nvfortran.

```bash
nvfortran -lcufft -cudalib=cufft -acc -Minfo=accel -o cufft.acc cufft_acc.f90
```
Here the flag `-acc` enables OpenACC on NVIDIA-GPU. Here it is possible to specify the compute capability in Betzy, e.g. `-acc=cc80`

To run:
```bash
srun --partition=accel --gpus=1 --time=00:01:00 --account=nnXXXXX --qos=devel --mem-per-cpu=1G ./cufft.acc
```

For completeness, porting the FFTW library to cuFFTW is straightforward: it is done by replacing….  
   
For an NVIDIA GPU-based case, the linking should be provided for both cuFFT and cuFFTW libraries (i.e. `-cudalib=cufft`).

# Conclusion

In conclusion, we have provided a short description on the implementation of the FFTW library in a serial code and the GPU-accelerated FFT targeting NVIDIA in an OpenACC application. Although the implementation has been shown for a 1D problem, an extension to 2D and 3D scenarios is straightforward.   

