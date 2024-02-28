# Image Analysis and Pattern Recognition

## General information
* Lecturer: [Jean-Philippe Thiran][jpt]
* [EE-451 coursebook][coursebook]
* [Moodle][moodle]

[moodle]: https://moodle.epfl.ch/course/view.php?id=5091
[jpt]: https://people.epfl.ch/115534
[coursebook]: https://edu.epfl.ch/coursebook/en/image-analysis-and-pattern-recognition-EE-451

### Objective of this course
Learn the basic methods of digital image analysis and pattern recognition:
pre-processing, image segmentation, shape representation and classification.
These concepts will be illustrated by applications in computer vision and
medical image analysis.

### Objective of this repository
This repository contains the material for the labs and project associated with
the [EPFL] master course
[EE-451 Image Analysis and Pattern Recognition][edu].

[epfl]: https://www.epfl.ch/
[edu]: https://edu.epfl.ch/coursebook/en/image-analysis-and-pattern-recognition-EE-451

## Installation instructions

### Python and packages
We will be using [git], [Python], and packages from the
[Python scientific stack][scipy].
If you don't know how to install these on your platform, we recommend to
install [Anaconda] or [Miniconda], both are distributions of the [conda]
package and environment manager.
Please follow the below instructions to install it and create an environment
for the course:

1. Download the latest Python 3.x installer for Windows, macOS, or Linux from
   <https://www.anaconda.com/download> or <https://conda.io/miniconda.html>
   and install with default settings.
   Skip this step if you have conda already installed (from [Miniconda] or
   [Anaconda]).
   Linux users may prefer to use their package manager.
   * Windows: Double-click on the `.exe` file.
   * macOS: Double-click on the `.pkg` file.
   * Linux: Run `bash Anaconda3-latest-Linux-x86_64.sh` in your terminal.
1. Open a terminal. For Windows: open the Anaconda Prompt from the Start menu.
1. Install git if not already installed with `conda install git`.
1. Download this repository by running
   `git clone https://github.com/LTS5/iapr`.
1. It is recommended to create an environment dedicated to this course with
   `conda create -n iapr python=3.9.18`.
1. Activate the environment:
   * Linux/macOS: `conda activate iapr`.
   * Windows: `activate iapr`.
1. Install the packages we will be using for this course:
   * `conda install notebook`
1. You can deactivate the environment whenever you are done with `deactivate`
   
[git]: https://git-scm.com
[python]: https://www.python.org
[scipy]: https://www.scipy.org
[anaconda]: https://anaconda.org
[miniconda]: https://conda.io/miniconda.html
[conda]: https://conda.io

### Python editors

[Jupyter] is a very useful editor that run directly in a web browser.
You can have Python and Markdown cells, hence it is very useful for
 examples, tutorials and labs.
 We encourage to use it for the lab assignments.
 To launch a Jupyter Notebook:
 * Open a terminal (for Windows open the Anaconda Prompt)
 * Move to the git repository `/path/to/iapr`
 * Activate the environment with `source activate iapr` (For Windows:
 `activate iapr`)
 * Run the command: `jupyter notebook`.

[jupyter]: https://jupyter.org/
