# PyTorch Deep Learning Notes

This project aims to systematically organize and document my experiences and insights gained during deep learning research and practice using PyTorch. Based on Professor Li Mu's "Dive into Deep Learning" course, I have progressively implemented and validated various models and algorithms covered in the course, from fundamental models to cutting-edge technologies, all with detailed code implementations and explanations.

---

## Project Background

Throughout my continuous exploration of deep learning, I have gradually realized the crucial importance of the mutual validation between theory and practice. To this end, I have organized full-pipeline practical cases covering data processing, model construction, training, and evaluation, including classic models (such as linear regression, multilayer perceptrons, convolutional neural networks), sequence models (such as RNN, GRU, LSTM, bidirectional RNN), and the latest Transformer and BERT models.

Furthermore, considering potential incompatibility issues arising from d2l library version updates, I have independently implemented numerous utility functions, all consolidated in the `utils.py` file. These utility functions not only improve code stability and reusability but also facilitate a deeper understanding of underlying implementation details.

---

## Directory Structure

- **Fundamental Modules**
  - Activation function implementations
  - Data preprocessing and dataset construction

- **Traditional Models**
  - Linear regression (basic and advanced implementations)
  - Multilayer perceptron (MLP)
  - Convolutional neural networks (CNN) and related regularization techniques

- **Sequence Models**
  - Basic RNN
  - GRU and LSTM
  - Deep RNN and bidirectional RNN

- **Advanced Models**
  - Machine translation and dataset construction
  - Encoder-decoder and Seq2Seq models
  - Attention mechanisms (including Bahdanau attention, multi-head attention, self-attention, and positional encoding)
  - Transformer and BERT pretraining and dataset construction

- **Utility Functions**
  - `utils.py`: To address issues arising from d2l library version updates, I have independently implemented a series of utility functions and integrated them here to ensure the project runs smoothly across different environments.

- **Experiments and Applications**
  - House price prediction case study
  - Other practical cases

---

## Runtime Environment

- **Development Tool**: PyCharm
- **Python**: 3.8+
- **PyTorch**: 2.0.0+cu117

### Installing PyTorch

Please use the following command to install PyTorch and related packages:

```bash
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html --timeout 1000
```

---

## Usage Instructions

Each script in the project is an independent module. It is recommended to read and experiment in sequence, starting with the foundational content and gradually advancing to advanced models and applications. Each module includes detailed comments and explanations to facilitate understanding of the principles and details of each implementation step.

---

## Acknowledgments

I am grateful to Professor Li Mu and the "Dive into Deep Learning" course for providing systematic theoretical and practical guidance, which has inspired my strong interest in deep learning and motivated my continuous exploration. I also thank the numerous contributors in the open-source community for sharing their experiences and code, from which I have greatly benefited throughout my learning journey.

---

## Contributions and License

Suggestions and improvements to this project are welcome. The project currently follows the MIT License, and any form of communication and feedback is appreciated.

---

This project is not only a summary of my personal learning journey but also an in-depth practice of deep learning research methodologies. I hope it will be beneficial to fellow practitioners.
