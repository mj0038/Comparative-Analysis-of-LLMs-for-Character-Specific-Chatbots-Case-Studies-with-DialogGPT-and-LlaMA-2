# Comparative Analysis of LLMs for Character-Specific Chatbots: Case Studies with DialogGPT and LlaMA-2

Welcome to our research project focusing on a comparative analysis between two prominent large language models, DialogGPT and LlaMA-2, specifically applied to character-specific chatbots. Our study delves into key metrics such as perplexity, BLEU scores, fluency, and relevancy to provide a comprehensive understanding of the performance of these models.

## Project Setup

### Dependencies Installation
Before running the project, ensure that all dependencies are installed. Execute the following command:

```bash
pip install -r requirements.txt
```

### Folder Structure
The project is organized into two main folders: DialoGPT and LlaMA-2. Each folder contains subfolders for source code (`src`), datasets (`dataset`), and model files (`model`).

## Running the Project

### Trained Models
#### DialoGPT
As the DialoGPT model files are substantial in size, they are not included in this repository. Please connect to Google Drive and use the provided links below to access the necessary model files:

- [**Big Bang Theory Model**](https://drive.google.com/drive/folders/1CBpYKxK1L1odNu6GLvd0RVuUtQa_K5Lb?usp=sharing)
- [**Rick and Morty Model**](https://drive.google.com/drive/folders/1BxKQ41QSZ6HOjCn0NVoerrzZhDKSfSse?usp=sharing)

Ensure that the "output-small" file is placed in the same directory as the `chat_with_bot.py` script.

### Execution Steps

1. To run the trained models, execute the `chat_with_bot.py` script in either the DialoGPT folder for either Rick and Morty or Big Bang Theory.
2. If running the entire code from scratch, ensure that your GPU is activated, then execute the `main.py` file.
3. Repeat the above steps for both the Rick and Morty and Big Bang Theory chatbots.

#### Llama-2
As the Llama-2 model files are substantial in size, they are not included in this repository. The models are loaded from huggingface ai. All the repository names are stores in their respective config.py files.  The links to the fine tuned models are as follows - 

- [**Big Bang Theory Model**](https://huggingface.co/rishabh27s/sheldon-llama2-chat)
- [**Rick and Morty Model**](https://huggingface.co/rishabh27s/rick-llama2-chat)

### Execution Steps


1. The Llama-2 folder has two subfolders for Big Bang Theory and Rick and Morty. Each of these files have their separate requirements, model_training, chat_with_bot and eval files. To run the code execute the following command - 
```bash
pip install -r requirements.txt
```

```bash
python3 main.py
```
2. If you want to train your own model selct option 1.
3. For chatting with the bot and evaluating the models ensure that you follow the instructions on the screen to see if you want to test you custom model or the fine-tuned model from my repository.
4. Ensure that you have enough GPU RAM hardware to run the code, It is quite exhaustive. A Tesla T4 GPU should work for all operations.

## Results and Evaluation

The results are presented in a comprehensive table, showcasing perplexity and BLEU scores for both models. Additionally, pie charts visually represent fluency and relevancy based on manual evaluations from a population of 500 people.

We hope this research contributes valuable insights to the field of character-specific chatbots, aiding developers and researchers in making informed decisions about model selection for their applications. Feel free to explore our findings and reach out for any further inquiries or discussions.
