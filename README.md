# Chatbot with Databricks Dolly Dataset and T5 Transformer

This repository contains code for building a chatbot using the Databricks Dolly dataset and the T5 transformer model. The chatbot is capable of providing responses based on pre-defined instructions and generating responses using the T5 model for unseen inputs.

## Features

- Load the Databricks Dolly dataset from a JSONL file
- Use the T5 transformer model for instruction-based responses
- Generate responses for unseen inputs using the T5 model
- Interactive chatbot functionality with user input and corresponding chatbot responses

## Requirements

- TensorFlow
- Transformers library
- Hugging Face Transformers model and tokenizer (T5)

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the script `chatbot.py`.
3. Enter your questions or instructions.
4. The chatbot will provide responses based on the available instructions in the dataset.
5. For unseen inputs, it will generate responses using the T5 model.

Feel free to customize the code to suit your specific requirements and dataset. Contributions and improvements are also welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
