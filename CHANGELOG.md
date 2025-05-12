# CHANGELOG

## Versioning Guide
0.0.**x** - Patch or minor functionality improvement(s)  
0.**x**.0 - Major reworking of the code or functionality improvement(s)  
**x**.0.0 - Completed development of intended project scope

## Current Development

**0.10.1**  NRMSE & per-m/z ERD metrics, OOM patches, centroid-only loading, deprecate .tsf, OpenTIMS replaces Alphatims, DESI-CSV format
**0.10.2**  Generative Learning Adversarial Network for Dynamic Sampling (GLANDS) framework  
**0.10.3**  Use defaults with overrides for custom configurations

## Future Development

**x.x.x**  Distributed data parallelism for multi-GPU training  
**x.x.x**  Scanning path optimization  
**x.x.x**  Integration of automatic m/z selection by IFS  
**x.x.x**  Alternate structurally-based evaluation metric  
**x.x.x**  Experimental MALDI integration  
**x.x.x**  Implementation of line revisiting in DESI  
**x.x.x**  Python package manager installation  

## Prior Releases

**0.10.0**  PyTorch migration, patches for DESI import speed, parallel visualizations, and model saving/loading  
**0.9.7**   Reload option for .hdf5 data, patch pointwise selection  
**0.9.6**   Adjust ERD processing and visualizations, add progression map, compute times, and alphatims option  
**0.9.5**   MALDI optical, input config, .tdf/.tsf compatibility, disable whole spectra metrics option, .imzML output  
**0.9.4**   Fix group-based and c value selection, distributed multi-GPU simulations, updated Otsu segmented linewise  
**0.9.3**   Whole spectra metrics, improved data aug. and file loading, fix RAM OOM, .imzML out, I/O norm. options  
**0.9.2**   .imzML, Bruker .d, image support, RD speedup, fix RD times, single sample training, FOV mask support  
**0.9.1**   Parallel sample loading, unique model names, post-processing mode, replace avg. m/z with TIC  
**0.9.0**   Multichannel E/RD, distributed GPU/batch training, E/RD timing, fix seq. runs  
**0.8.9**   Simplification  
**0.8.8**   Interpolation limits, static graph, parallel inferencing, ray deployment, test of FAISS  
**0.8.7**   Recon. script, Options for acq. rate, sequential names, live output, row offsets, and input scaling  
**0.8.6**   Memory reduction, m/z reconstruction vectorization, augmentation, global m/z, m/z window in ppm  
**0.8.5**   Model optimization, enable batch processing, SLADS training fix, database acceleration  
**0.8.4**   Parallel c value selection fix, remove network resizing requirement, fix experimental  
**0.8.3**   Mask seed fix, normalization for sim. fix, non-Ray option, pad instead of resize  
**0.8.2**   Multichannel, fixed groupwise, square pixels, accelerated RD, altered visuals/metrics  
**0.8.1**   Model simplification, method cleanup, m/z tolerance/standard patch  
**0.8.0**   Raw MSI file integration (.raw, .d), .d files only Windows compatible  
**0.6.9**   Do not use -- Original SLADS(-Net) variations for comparison with 0.7.3  
**0.7.5**   c value selection performed before training database generation  
**0.7.4**   CPU compatibility patch, removal of NaN values  
**0.7.3**   Start/End line patch, SLADS(-Net) options, normalization optimization  
**0.7.2**   Remove custom pkg. dependency, use NN resize, recon+measured input  
**0.7.1**   c value selection performed before model training  
**0.7.0**   CNN/U-Net/RBDN with dynamic window size  
**0.6.8**   Fixed RD generation, added metrics, and Windows compatible  
**0.6.7**   Clean asymmetric implementation with density features  
**0.6.6**   SLADS-Net NN, PSNR, asymFinal, and multi-config  
**0.6.5**   Clean variables and resize to physical  
**0.6.4**   Custom knn metric, SSIM calc, init computations  
**0.6.3**   Start/End point selection with Canny  
**0.6.2**   Model loading and animation production patches  
**0.6.1**   Model robustness and reduction of memory overhead  
**0.6.0**   Modifications for Nano-DESI microscope integration  
**0.5.0**   Overhead reduction; switch multiprocessing package  
**0.4.0**   Class/function segmentation  
**0.3.0**   Complete code rewrite, computational improvements  
**0.2.0**   Line bounded constraints addition  
**0.1.1**   Line constraints, concatenation, pruning, and results organization    
**0.1.0**   Multithreading adjustments to pointwise SLADS  