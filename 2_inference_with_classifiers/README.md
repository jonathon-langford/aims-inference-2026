# 2. Inference with classifiers
Session leader: Jon Langford

This lecture will demonstrate how to use a simple machine learning (ML) classifier for simulation-based inference (SBI). This will help you bridge the gap between ML and some fundamental concepts in statistics.

The slides for this lecture are available [here](lecture_inference_with_classifiers.pdf).

We will cover the following topics:
- Brief review of frequentist inference techniques
- Introduction to intractable likelihoods and the need for SBI in the modern era of science.
- Understand how to use a simple ML classifier to learn the (log)-likelihood ratio
- Apply this technique to perform a hypothesis test for a research problem with an unknown likelihood
- Extend to parameter estimation by learning the conditional likelihood ratio with a parametric classifier. Compare the performance to the analytic solution for a simple 2D Gaussian example.

I hope you enjoy this dive into SBI with classifiers! For me it's very interesting to see how such a simple concept in ML can be used for a complex task like inference. If you have any questions or want to discuss the material further, please don't hesitate to reach out. Either chat to me in person this week or (if your interests are more long-term) send me an email: j.langford17@imperial.ac.uk

## Accompanying notebooks
The best way to understand the material covered in the lecture is to work through the accompanying notebooks. The notebooks are stored in the [notebooks](notebooks/) directory of this repository. 

The easiest way to run the notebooks is to use Google Colab. Just click on the buttons below...

- Hypothesis test (simple 1D Gaussian) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathon-langford/aims-inference-2026/blob/main/2_inference_with_classifiers/notebooks/hypothesis_test_simple.ipynb)

- Hypothesis test (particle spin example) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathon-langford/aims-inference-2026/blob/main/2_inference_with_classifiers/notebooks/hypothesis_test_particle_spin.ipynb)

- Parameter estimation (2D Gaussian) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonathon-langford/aims-inference-2026/blob/main/2_inference_with_classifiers/notebooks/parameter_estimation.ipynb)

If you prefer to run the notebooks locally, you can do so by first cloning this repository, navigating to the [notebooks](notebooks/) directory, setting up a Python environment with the required dependencies, and then running the notebooks using Jupyter Notebook or Jupyter Lab.

```bash
git clone https://github.com/jonathon-langford/aims-inference-2026.git
cd aims-inference-2026/2_inference_with_classifiers/notebooks

# Set up a Python environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch
```

Then run the notebooks using your preferred Jupyter interface (I find VS Code's Jupyter extension to be very convenient for this).
