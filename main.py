import os
import yaml
import json
import argparse
from langchain_core.load import dumps
from langchain_google_genai import ChatGoogleGenerativeAI


def setup(credentials_path: str = ".credentials.yaml"):
    with open(credentials_path, 'r') as file:
        credentials = yaml.safe_load(file)

    os.environ['GEMINI_API_KEY'] = credentials['gemini']['api_key']
    os.environ['LANGSMITH_API_KEY'] = credentials['langsmith']['api_key']

    return


def get_llm(model_config_yaml: str = "model_config.yaml", model_type: str = "gemini"):
    with open(model_config_yaml, 'r') as file:
        model_configs = yaml.safe_load(file)

    model = ChatGoogleGenerativeAI(
        model=model_configs[model_type]['model'],
        max_tokens=model_configs[model_type]['max_tokens'],
        temperature=model_configs[model_type]['temperature'],
        timeout=model_configs[model_type]['timeout'],
        max_retries=model_configs[model_type]['max_retries']
    )

    return model


def test_model(model):
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        (
            "human",
            "I love programming."
        ),
    ]
    ai_msg: json = model.invoke(messages)
    print(ai_msg)
    print("Text: ", ai_msg.text)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM with specified configurations.")
    parser.add_argument('--credentials', type=str, default=".credentials.yaml", help='Path to credentials YAML file.')
    parser.add_argument('--model_config', type=str, default="model_config.yaml", help='Path to model configuration YAML file.')
    parser.add_argument('--model_type', type=str, default="gemini", help='Type of model to use (e.g., gemini).')
    parser.add_argument('--test', action='store_true', help='Run test after setup.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    setup(credentials_path=args.credentials)
    llm = get_llm(model_config_yaml=args.model_config, model_type=args.model_type)

    if args.test:
        test_model(llm)
