# PathwayAnalysis

PathwayAnalysis is a Python library designed to do the following two tasks.
* Compute pathway expression data from gene expression data. See the in review paper [Pathway Expression Analysis](https://assets.researchsquare.com/files/rs-1981270/v1_covered.pdf?c=1661534668)
* Identify small discriminatory gene co-expression modules using spectral hierarchical clustering using a modified version of the iterative spectral clustering algorithm in [Methods for Network Generation and Spectral Feature Selection: Especially on Gene Expression Data](https://www.proquest.com/docview/2378897983?pq-origsite=gscholar&fromopenview=true)

## How to Cite

If you like the pathway expression code, please consider citing us.
```
@article{mankovich2022pathway,
  title={Pathway expression analysis},
  author={Mankovich, Nathan and Kehoe, Eric and Peterson, Amy and Kirby, Michael},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={21839},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```


If you like the spectral clustering code, please consider citing us.
```
@phdthesis{mankovich2019methods,
  title={Methods for Network Generation and Spectral Feature Selection: Especially on Gene Expression Data},
  author={Mankovich, Nathan},
  year={2019},
  school={Colorado State University}
}
```


## Installation

To install PathwayAnalysis, python version 3.8 or greater is required. All other required packages and versions can be found in requirements.txt.

### Install with the most recent commits

git cloning the [PathwayAnalysis](https://github.com/nmank/PathwayAnalysis), going to the PathwayAnalysis directory, run

`pip install -e .`

## Tutorials

See the jupyter notebooks in the folder ./examples

