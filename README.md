DivMBest
========

This is the source release accompanying the following various publications:

    @inproceedings{divmbest_ECCV2012,
        Author = {Dhruv Batra, Payman Yadollahpour, Abner Guzman-Rivera, Greg Shakhnarovich}
        Title = {Diverse M-Best Solutions in Markov Random Fields}
        Booktitle = {European Conference on Computer Vision (ECCV)}
        year = {2012}
    }
  
  
    @inproceedings{embr_CVPR2014,
        Author = {Vittal Premachandran, Daniel Tarlow and Dhruv Batra},
        Title = {Empirical Minimum Bayes Risk Prediction: How to extract an extra few % performance from vision models with just three more parameters}
        Booktitle = {CVPR},
        Year = {2014},
    }

## Abstract

> Much effort has been directed at algorithms for obtaining the highest probability (MAP) configuration in probabilistic (random
field) models. In many situations, one could benefit from additional high-probability solutions. Current methods for computing the M most probable configurations produce solutions that tend to be very similar to the MAP solution and each other. This is often an undesirable property. In this paper we propose an algorithm for the Diverse M-Best problem, which involves finding a diverse set of highly probable solutions under a discrete probabilistic model. Given a dissimilarity function measuring
closeness of two solutions, our formulation involves maximizing a linear combination of the probability and dissimilarity to previous solutions. Our formulation generalizes the M-Best MAP problem and we show that for certain families of dissimilarity functions we can guarantee that these solutions can be found as easily as the MAP solution.

##Instructions

Please move into each directory and read the respective README files for instructions on how to use the code.

### Code Contributors

[Abner Guzman-Rivera](http://abnerguzman.com/), [Vittal Premachandran](http://www.comp.nus.edu.sg/~vittal/), [Senthil Purushwalkam](https://github.com/senthilps8), [Payman Yadollahpour](http://ttic.uchicago.edu/~pyadolla/)
