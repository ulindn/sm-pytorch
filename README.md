## PyTorch models using SageMaker in local environment

This example uses [PyTorch quick start tutotial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) and trains a model using [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/index.html) as in [Amazon SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples/tree/main/frameworks/pytorch) using [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The model can be trained in two ways: by running the prebuilt AWS SageMaker Pytorch container in AWS or by running it locally. For inference, the model is tested locally. 

The development environment used was:
- Ubuntu with Docker Compose v2
- VSCode with conda 
